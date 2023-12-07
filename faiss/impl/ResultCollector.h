/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_set>
#include <vector>

#include <faiss/MetricType.h>
#include <faiss/utils/Heap.h>

/**
 * ResultCollector is intended to define how to collect search result
 * For each single search result, collect method will be called.
 * After every results are collected, post_process method is called at the end.
 */

namespace faiss {

/** Encapsulates a set of ids to handle. */
struct ResultCollector {
    /**
     * For each result, collect method is called to store result
     * @param k         number of vectors to search
     * @param nres      number of results in queue
     * @param bh_val    search result, distances from query
     * @param bh_ids    search result, ids of vectors
     * @param val       distance from query for current vector
     * @param ids       id of current vector
     */
    virtual void collect(
            int k,
            int& nres,
            float* bh_val,
            idx_t* bh_ids,
            float val,
            idx_t ids) = 0;

    // This method is called after all result is collected
    virtual void post_process(idx_t nres, idx_t* bh_ids) = 0;
    virtual ~ResultCollector() {}
};

struct DefaultCollector : ResultCollector {
    void collect(
            int k,
            int& nres,
            float* bh_val,
            idx_t* bh_ids,
            float val,
            idx_t ids) override {
        if (nres < k) {
            faiss::maxheap_push(++nres, bh_val, bh_ids, val, ids);
        } else if (val < bh_val[0]) {
            faiss::maxheap_replace_top(nres, bh_val, bh_ids, val, ids);
        }
    }

    // This method is called once all result is collected so that final post
    // processing can be done For example, if the result is collected using
    // group id, the group id can be converted back to its original id inside
    // this method
    void post_process(idx_t nres, idx_t* bh_ids) override {
        // Do nothing
    }

    ~DefaultCollector() override {}
};

} // namespace faiss
