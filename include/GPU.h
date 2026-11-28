#ifndef GPU_H_
#define GPU_H_

#include <string>
#include <cstddef>
#include <vector>
#include <utility>
class Request;
class GenerativeRequest;
class EmbeddingRequest;
class MemoryBlock;

struct PrefixMatch {
    std::size_t matchedTokens;
    std::vector<std::size_t> blockIds;
};

class MemoryBlock;

class GPU{
public:
    //rule of five
    ~GPU();
    GPU(const GPU& other);
    GPU& operator=(const GPU& other);
    GPU(GPU&& other) noexcept;
    GPU& operator=(GPU&& other) noexcept;
    //end of rule of five
    GPU(std::size_t blocksNum, std::size_t blockSize);
    std::size_t availableSlots() const;
    std::size_t getBlocksForMemory(std::size_t memory) const;
    void reserve(std::size_t n, const Request& request);
    void unreserve(std::size_t n, const Request& request);
    std::size_t allocateBlock(const Request& request);
    void addMemoryBlock(const GenerativeRequest& request, std::size_t slot);
    void addMemoryBlock(const EmbeddingRequest& request, std::size_t slot);
    void acquire(std::size_t slot);
    void release(std::size_t slot);
    void copyMemoryFromOtherGPU(const GPU& other,const std::vector<std::size_t>& blocks);
    void assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId);
    int readFromBlock(std::size_t blockId, std::size_t offset) const;
    void writeToBlock(std::size_t blockId, std::size_t offset, int value);
    PrefixMatch findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                        const std::vector<std::string>& promptTokens) const;
    std::size_t getBlockSize() const ;
private:
    std::size_t blocksNum;
    std::size_t blockSize;
    std::size_t reserved;
    std::vector<MemoryBlock*> memory;
    std::vector<std::pair<const Request*, std::size_t>> reservations;
    //for prefix matching
    std::size_t evicted;
    std::vector<std::vector<std::string>> prefixKeys;
    std::vector<std::string> blockModelIds;
    std::vector<bool> hasPrefix;
};

#endif