#include"SchedulingPolicy.h"
#include"GPU.h"
#include"Request.h"
#include"Model.h"
#include <algorithm>


//ArrivalOrderPolicy
std::vector<int> ArrivalOrderPolicy::choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const {
    std::vector<int> result;
    for (const Request* request : eligible) {
        result.push_back(request->getId());
    }
    return result;
}

SchedulingPolicy* ArrivalOrderPolicy::clone() const {
    return new ArrivalOrderPolicy(*this);
}

//ShortestJobPolicy
std::vector<int> ShortestJobPolicy::choose(const std::vector<const Request*>& eligible,const std::vector<GPU>& gpus) const {

    std::vector<int> result;

    for (std::size_t i = 0; i < eligible.size(); ++i) {
        result.push_back(i);
    }

    std::sort(result.begin(), result.end(),[&](int a, int b) {

            std::size_t blocksA =gpus[0].getBlocksForMemory(eligible[a]->getMaxNeededMemory());

            std::size_t blocksB =gpus[0].getBlocksForMemory(eligible[b]->getMaxNeededMemory());

            if (blocksA != blocksB) {
                return blocksA < blocksB;
            }

            return eligible[a]->getId() < eligible[b]->getId();
        }
    );

    return result;
}

SchedulingPolicy* ShortestJobPolicy::clone() const {
    return new ShortestJobPolicy(*this);
}

//LongestPrefixPolicy
std::vector<int> LongestPrefixPolicy::choose(const std::vector<const Request*>& eligible,const std::vector<GPU>& gpus) const {

    std::vector<int> result;

    for (std::size_t i = 0; i < eligible.size(); ++i) {
        result.push_back(i);
    }

    std::sort(result.begin(), result.end(),[&](int a, int b) {

            std::size_t prefixA = 0;
            std::size_t prefixB = 0;

            for (const GPU& gpu : gpus) {

                PrefixMatch matchA = gpu.findLongestCachedPrefix(eligible[a]->getModel().getId(),eligible[a]->getModel().getEmbeddingSize(),eligible[a]->getCacheableTokens());

                PrefixMatch matchB = gpu.findLongestCachedPrefix(eligible[b]->getModel().getId(),eligible[b]->getModel().getEmbeddingSize(),eligible[b]->getCacheableTokens());

                if (matchA.matchedTokens > prefixA) {
                    prefixA = matchA.matchedTokens;
                }

                if (matchB.matchedTokens > prefixB) {
                    prefixB = matchB.matchedTokens;
                }
            }

            if (prefixA != prefixB) {
                return prefixA > prefixB;
            }

            return eligible[a]->getId() < eligible[b]->getId();
        }
    );

    return result;
}

SchedulingPolicy* LongestPrefixPolicy::clone() const {
    return new LongestPrefixPolicy(*this);
}
