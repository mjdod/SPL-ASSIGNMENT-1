#include "Request.h"
#include "Model.h"
#include "GPU.h"

Request::Request(int id, const Model& model, int admissionCycle, std::string prompt):
    id(id), model(model), admissionCycle(admissionCycle), tokens(model.tokenizer(prompt)), memoryBlocks(), reservedBlocks(0)
{}

void Request::deallocateMemory(GPU& gpu) {
    gpu.unreserve(getNeedToUnreserve(), *this);

    for (std::size_t slot : memoryBlocks) {
        gpu.release(slot);
    }

    memoryBlocks.clear();
    reservedBlocks = 0;
}

int Request::getId() const {
    return id;
}

const Model& Request::getModel() const {
    return model;
}

int Request::getAdmissionCycle() const {
    return admissionCycle;
}
//GenerativeRequest
GenerativeRequest::GenerativeRequest(int id,const Model& model,int admissionCycle,std::string prompt,std::string scriptedOutput,std::size_t maxOutputTokens)
    : Request(id, model, admissionCycle, prompt),scriptedOutput(scriptedOutput),maxOutputTokens(maxOutputTokens),outputTokens(model.tokenizer(scriptedOutput)),processedTokens(0) {

    if (outputTokens.size() > maxOutputTokens) {
        outputTokens.resize(maxOutputTokens);
    }
}

std::size_t GenerativeRequest::getMaxNeededMemory() const{
    return model.getEmbeddingSize() *(tokens.size() + maxOutputTokens);;
}
std::size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const {

    PrefixMatch match = gpu.findLongestCachedPrefix(model.getId(),model.getEmbeddingSize(),tokens);

    return model.getEmbeddingSize() * (tokens.size() - match.matchedTokens + maxOutputTokens);
}
std::vector<std::string> GenerativeRequest::getCacheableTokens() const {
    return tokens;
}
void GenerativeRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const {
     gpu.addMemoryBlock(*this, slot);
}
Request* GenerativeRequest::clone(const Model& model) const {

    std::string result;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            result += " ";
        }
        result += tokens[i];
    }

    return new GenerativeRequest(id, model, admissionCycle, result, scriptedOutput, maxOutputTokens);
}

void GenerativeRequest::act(GPU& gpu) {

    if (finished()) {
        return;
    }

    std::size_t embeddingSize = model.getEmbeddingSize();
    std::size_t tokensPerBlock = gpu.getBlockSize() / embeddingSize;

    // Build the sequence up to the current token
    std::vector<std::string> prefixSeq;

    for (std::size_t i = 0; i <= processedTokens; ++i) {
        if (i < tokens.size()) {
            prefixSeq.push_back(tokens[i]);
        }
        else {
            prefixSeq.push_back(outputTokens[i - tokens.size()]);
        }
    }

    std::size_t blockIdx = processedTokens / tokensPerBlock;
    std::size_t offset = (processedTokens % tokensPerBlock) * embeddingSize;

    if (blockIdx == memoryBlocks.size()) {
        std::size_t slot = gpu.allocateBlock(*this);
        memoryBlocks.push_back(slot);
        --reservedBlocks;
    }

    model.generateTokenEmbedding(
        prefixSeq,
        gpu,
        memoryBlocks[blockIdx],
        offset
    );

    if (processedTokens + 1 <= tokens.size() &&
        (processedTokens + 1) % tokensPerBlock == 0) {

        gpu.assignPrefixToBlock(
            memoryBlocks[blockIdx],
            prefixSeq,
            model.getId()
        );
    }

    ++processedTokens;
}

bool GenerativeRequest::finished() const {
    return processedTokens >= tokens.size() + outputTokens.size();
}
void GenerativeRequest::reserveMemory(GPU& gpu) {

    PrefixMatch match = gpu.findLongestCachedPrefix(model.getId(),model.getEmbeddingSize(),tokens);

    for (std::size_t slot : match.blockIds) {
        gpu.acquire(slot);
        memoryBlocks.push_back(slot);
    }

    processedTokens = match.matchedTokens;

    std::size_t neededMemory = getNeededMemory(gpu);

    reservedBlocks = gpu.getBlocksForMemory(neededMemory);

    gpu.reserve(reservedBlocks, *this);
}

std::string GenerativeRequest::getOutput() const {
    std::string out;
    for (std::size_t i = 0; i < outputTokens.size(); ++i) {
        if (i > 0) {
            out += " ";
        }
        out += outputTokens[i];
    }
    return out;
}
std::size_t GenerativeRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}

//EmbeddingRequest
EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt)
    : Request(id, model, admissionCycle, prompt),processedTokens(0) {}

std::size_t EmbeddingRequest::getMaxNeededMemory() const {
    return 2 * model.getEmbeddingSize();
}

std::size_t EmbeddingRequest::getNeededMemory(const GPU& gpu) const {
        return 2 * model.getEmbeddingSize();
}

std::vector<std::string> EmbeddingRequest::getCacheableTokens() const {
    return {};
}

void EmbeddingRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const {
    gpu.addMemoryBlock(*this, slot);
}

Request* EmbeddingRequest::clone(const Model& model) const {
    std::string result;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            result += " ";
        }
        result += tokens[i];
    }
    return new EmbeddingRequest(id, model, admissionCycle, result);
}

void EmbeddingRequest::act(GPU& gpu) {
    if (finished()) {
        return;
    }

    const std::size_t embeddingSize = model.getEmbeddingSize();
    const std::size_t blockSize = gpu.getBlockSize();

    const std::size_t requiredBlocks =
        gpu.getBlocksForMemory(2 * embeddingSize);

    while (memoryBlocks.size() < requiredBlocks) {
        std::size_t slot = gpu.allocateBlock(*this);
        memoryBlocks.push_back(slot);
        --reservedBlocks;
    }

    std::size_t sumSlot = memoryBlocks[0];

    std::size_t currentSlot;
    std::size_t currentOffset;

    if (blockSize >= 2 * embeddingSize) {
        currentSlot = memoryBlocks[0];
        currentOffset = embeddingSize;
    } else {
        currentSlot = memoryBlocks[1];
        currentOffset = 0;
    }

    model.generateTokenEmbedding(
        {tokens[processedTokens]},
        gpu,
        currentSlot,
        currentOffset
    );

    for (std::size_t i = 0; i < embeddingSize; ++i) {
        int sum = gpu.readFromBlock(sumSlot, i);
        int current = gpu.readFromBlock(currentSlot, currentOffset + i);

        gpu.writeToBlock(sumSlot, i, sum + current);
    }

    ++processedTokens;
}

bool EmbeddingRequest::finished() const {
    return processedTokens >= tokens.size();
}

bool EmbeddingRequest::finished() const {
   return processedTokens >= tokens.size();
}
void EmbeddingRequest::reserveMemory(GPU& gpu) {

    reservedBlocks = gpu.getBlocksForMemory(2 * model.getEmbeddingSize());

    gpu.reserve(reservedBlocks, *this);
}
std::size_t EmbeddingRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}