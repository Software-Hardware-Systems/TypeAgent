// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import { DateTimeRange } from "./dateTimeSchema.js";

export type FacetTerm = {
    // the name of the facet, such as "color", "profession", "patent number"; "*" means match any facet name
    facetName: string;
    // the value of the facet, such as "red", "writer"; "*" means match any facet value
    facetValue: string;
};

// Use to find information about specific, tangible people, places, institutions or things only..
// This includes entities with particular facets
// Abstract concepts or topics are not entities.
// Any terms will match fuzzily.
export type EntityTerm = {
    // the name of the entity or thing such as "Bach", "Great Gatsby", "frog" or "piano" or "we", "I"; "*" means match any entity name
    name: string;
    isNamePronoun: boolean;
    // the specific types of the entity such as "book", "movie", "song", "speaker", "person", "artist", "animal", "instrument", "school", "room", "museum", "food" etc.
    // Do not include generic types such as: "entity", "object", "thing", "concept" etc.
    // An entity can have multiple types; entity types should be single words
    type?: string[];
    // Facet terms search for properties or attributes of the entity.
    // Eg: color(blue), profession(writer), author(*), aunt(Agatha), weight(4kg), phoneNumber(...), title(*) etc.
    facets?: FacetTerm[];
};

export type VerbsTerm = {
    words: string[]; // individual words in single or compound verb
    tense: "Past" | "Present" | "Future";
};

export type ActionTerm = {
    // Action verbs describing the interaction
    actionVerbs?: VerbsTerm | undefined;
    // The origin of the action or information, typically the entity performing the action
    actorEntities: EntityTerm[] | "*";
    // the recipient or target of the action or information
    // Action verbs can imply relevant facet names on the targetEntity. E.g. write -> writer, sing -> singer etc.
    targetEntities?: EntityTerm[];
    // additional entities participating in the action.
    // E.g. in the phrase "Jane ate the spaghetti with the fork", "the fork" would be an additional entity
    // E.g. in the phrase "Did Jane speak about Bach with Nina", "Bach" would be the additional entity
    additionalEntities?: EntityTerm[];
    // Is the intent of the phrase translated to this ActionTerm to actually get information about a specific entities?
    // Examples:
    // true: if asking for specific information about an entity, such as "What is Mia's phone number?" or "Where did Jane study?"
    // false if involves actions and interactions between entities, such as "What phone number did Mia mention in her note to Jane?"
    isInformational: boolean;
};

export type ScopeFilter = {
    // Search terms to use to limit search
    entitySearchTerms?: EntityTerm[] | undefined;
    // Use only if request explicitly asks for time range, particular year, month etc.
    timeRange?: DateTimeRange | undefined; // in this time range
};

//
// Search a search engine using filters:
// entitySearchTerms cannot contain entities already in actionSearchTerms
export type SearchFilter = {
    actionSearchTerm?: ActionTerm;
    entitySearchTerms?: EntityTerm[];
    // searchTerms:
    // Use for all concepts, topics, or other search terms that don't fit ActionTerms or EntityTerms
    // - Do not use noisy searchTerms like "topic", "topics", "subject", "discussion" etc. even if they are mentioned in the user request
    // - Phrases like 'email address' or 'first name' are a single term
    // - use empty searchTerms array when use asks for summaries
    searchTerms?: string[];
    // Use to limit matches to particular sub-set of a conversation, document, etc
    // E.g. if the user request specifies a particular section or context, use scopeSubQuery to limit the search.
    scopeSubQuery?: ScopeFilter | undefined;
};

export type SearchExpr = {
    rewrittenQuery: string;
    // Translate rewritten query into filters
    filters: SearchFilter[];
};

export type SearchQuery = {
    // One expression for each search required by user request
    // Each SearchExpr runs independently, so make them standalone by resolving references like 'it', 'that', 'them' etc.
    searchExpressions: SearchExpr[];
};
