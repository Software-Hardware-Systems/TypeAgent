# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import asyncio
import builtins
from dataclasses import dataclass
import json

import numpy as np
import typechat

from typeagent.aitools import utils
from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.knowpro.answer_response_schema import AnswerResponse
from typeagent.knowpro import answers
from typeagent.knowpro.importing import ConversationSettings
from typeagent.knowpro.convknowledge import create_typechat_model
from typeagent.knowpro.interfaces import IConversation
from typeagent.knowpro.search_query_schema import SearchQuery
from typeagent.knowpro import searchlang
from typeagent.podcasts.podcast import Podcast


@dataclass
class Context:
    conversation: IConversation
    query_translator: typechat.TypeChatJsonTranslator[SearchQuery]
    answer_translator: typechat.TypeChatJsonTranslator[AnswerResponse]
    embedding_model: AsyncEmbeddingModel
    lang_search_options: searchlang.LanguageSearchOptions
    answer_options: answers.AnswerContextOptions
    interactive: bool


def main():
    # Parse arguments.

    default_qafile = "testdata/Episode_53_Answer_results.json"
    default_podcast_file = "testdata/Episode_53_AdrianTchaikovsky_index"

    explanation = "a list of objects with 'question' and 'answer' keys"
    parser = argparse.ArgumentParser(description="Parse Q/A data file")
    parser.add_argument(
        "--qafile",
        type=str,
        default=default_qafile,
        help=f"Path to the data file ({explanation})",
    )
    parser.add_argument(
        "--podcast",
        type=str,
        default=default_podcast_file,
        help="Path to the podcast index files (excluding the '_index.json' suffix)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of initial Q/A pairs to skip",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of Q/A pairs to print (0 means all)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        default=False,
        help="Run in interactive mode, waiting for user input before each question",
    )
    args = parser.parse_args()

    # Read evaluation data.

    with open(args.qafile, "r") as file:
        data = json.load(file)
    assert isinstance(data, list), "Expected a list of Q/A pairs"
    assert len(data) > 0, "Expected non-empty Q/A data"
    assert all(
        isinstance(qa_pair, dict) and "question" in qa_pair and "answer" in qa_pair
        for qa_pair in data
    ), "Expected each Q/A pair to be a dict with 'question' and 'answer' keys"

    # Read podcast data.

    utils.load_dotenv()
    settings = ConversationSettings()
    with utils.timelog("Loading podcast data"):
        conversation = Podcast.read_from_file(args.podcast, settings)
    assert conversation is not None, f"Failed to load podcast from {file!r}"

    # Create translators.

    model = create_typechat_model()
    query_translator = utils.create_translator(model, SearchQuery)
    answer_translator = utils.create_translator(model, AnswerResponse)

    # Create context.

    context = Context(
        conversation,
        query_translator,
        answer_translator,
        AsyncEmbeddingModel(),
        lang_search_options=searchlang.LanguageSearchOptions(
            compile_options=searchlang.LanguageQueryCompileOptions(
                exact_scope=False, verb_scope=True, term_filter=None, apply_scope=True
            ),
            exact_match=False,
            max_message_matches=25,
        ),
        answer_options=answers.AnswerContextOptions(
            entities_top_k=50, topics_top_k=50, messages_top_k=None, chunking=None
        ),
        interactive=args.interactive,
    )
    utils.pretty_print(context.lang_search_options)
    utils.pretty_print(context.answer_options)

    # Loop over eval data, skipping duplicate questions
    # (Those differ in 'cmd' value, which we don't support yet.)

    offset = args.offset
    limit = args.limit
    last_q = ""
    counter = 0
    all_scores: list[tuple[float, int]] = []  # [(score, counter), ...]
    for qa_pair in data:
        question = qa_pair.get("question")
        answer = qa_pair.get("answer")
        if question:
            question = question.strip()
        if answer:
            answer = answer.strip()
        if not (question and answer) or question == last_q:
            continue
        counter += 1
        last_q = question

        # Process offset if specified.
        if offset > 0:
            offset -= 1
            continue

        # Wait for user input before continuing.
        if context.interactive:
            try:
                input("Press Enter to continue... ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

        # Compare the given answer with the actual answer for the question.
        actual_answer, score = asyncio.run(compare(context, qa_pair))
        all_scores.append((score, counter))
        good_enough = score >= 0.97
        sep = "-" if good_enough else "*"
        print(sep * 25, counter, sep * 25)
        print(f"Score: {score:.3f}; Question: {question}", flush=True)
        if context.interactive or not good_enough:
            cmd = qa_pair.get("cmd")
            if cmd and cmd != f'@kpAnswer --query "{question}"':
                print(f"Command: {cmd}")
            if qa_pair.get("hasNoAnswer"):
                answer = f"Failure: {answer}"
            print(f"Expected answer:\n{answer}")
            print("-" * 20)
            print(f"Actual answer:\n{actual_answer}", flush=True)

        # Process limit if specified.
        if limit > 0:
            limit -= 1
            if limit == 0:
                break

    print("=" * 50)
    all_scores.sort(reverse=True)
    good_scores = [(score, counter) for score, counter in all_scores if score >= 0.97]
    bad_scores = [(score, counter) for score, counter in all_scores if score < 0.97]
    for label, pairs in [("Good", good_scores), ("Bad", bad_scores)]:
        print(f"{label} scores ({len(pairs)}):")
        for i in range(0, len(pairs), 10):
            print(
                ", ".join(
                    f"{score:.3f}({counter})" for score, counter in pairs[i : i + 10]
                )
            )


async def compare(
    context: Context, qa_pair: dict[str, str | None]
) -> tuple[str | None, float]:
    the_answer: str | None = None
    score = 0.0

    question = qa_pair.get("question")
    answer = qa_pair.get("answer")
    failed = qa_pair.get("hasNoAnswer")
    cmd = qa_pair.get("cmd")
    if not (question and answer):
        return None, score

    if not context.interactive:
        print = lambda *args, **kwds: None  # Disable printing in non-interactive mode
    else:
        print = builtins.print

    print()
    print("=" * 40)
    if cmd:
        print(f"Command: {cmd}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("-" * 40)

    result = await searchlang.search_conversation_with_language(
        context.conversation,
        context.query_translator,
        question,
        context.lang_search_options,
    )
    print("-" * 40)
    if not isinstance(result, typechat.Success):
        print("Error:", result.message)
    else:
        all_answers, combined_answer = await answers.generate_answers(
            context.answer_translator,
            result.value,
            context.conversation,
            question,
            options=context.answer_options,
        )
        print("-" * 40)
        if combined_answer.type == "NoAnswer":
            if failed:
                score = 1.0
            the_answer = f"Failure: {combined_answer.whyNoAnswer}"
            print(the_answer)
            print("All answers:")
            if context.interactive:
                utils.pretty_print(all_answers)
        else:
            assert combined_answer.answer is not None, "Expected an answer"
            the_answer = combined_answer.answer
            if failed:
                score = 0.0
            else:
                score = await equality_score(context, answer, the_answer)
            print(the_answer)
            print("Correctness score:", score)
    print("=" * 40)

    return the_answer, score


async def equality_score(context: Context, a: str, b: str) -> float:
    a = a.strip()
    b = b.strip()
    if a == b:
        return 1.0
    if a.lower() == b.lower():
        return 0.999
    embeddings = await context.embedding_model.get_embeddings([a, b])
    assert embeddings.shape[0] == 2, "Expected two embeddings"
    return np.dot(embeddings[0], embeddings[1])


if __name__ == "__main__":
    main()
