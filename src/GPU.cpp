#include <iostream>
#include "GPU.h"
#include "Request.h"
#include "MemoryBlock.h"
#include <stdexcept>
//constructer (still need to implment rule of 5)
GPU::GPU(std::size_t blocksNum, std::size_t blockSize): blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum,nullptr), reserved(0),reservations(),prefixKeys(blocksNum),blockModelIds(blocksNum),hasPrefix(blocksNum, false)
{
}



//the functions required in implementation O-O
std::size_t GPU::availableSlots() const
{
    int usedSlots = 0;
    for(MemoryBlock* i: memory)
    {
        if(i != nullptr && i->getRefCount()>0)
        {
            usedSlots++;
        }
    }
    return blocksNum - usedSlots - reserved;
}

std::size_t GPU::getBlocksForMemory(std::size_t memory) const 
{
    return (memory + blockSize - 1) / blockSize;
}
void GPU::reserve(std::size_t n, const Request& request)
{
    for (auto& r : reservations) {
        if (r.first == &request) {
            r.second += n;
            reserved += n;
            return;
        }
    }

    reservations.push_back({&request, n});
    reserved += n;
}
void GPU::unreserve(std::size_t n, const Request& request)
{
    for (auto& r : reservations)
    {
        if(r.first == &request)
        {
            if(n> r.second)
            {
                throw std::logic_error("reserved blocks are smaller than the ones i need to unreserve");
            }
            r.second -=n;
            reserved -= n;
            return;
        }
    }
    if (n != 0)
    {
        throw std::logic_error("request has nothing reserved");
    }
}

std::size_t GPU::allocateBlock(const Request& request)
{
    size_t slot = blocksNum;
    bool hasReservation = false;
    for (const auto& r : reservations)
    {
        if (r.first == &request && r.second > 0)
        {
            hasReservation = true;
            break;
        }
    }
    if (!hasReservation)
    {
        throw std::logic_error("request has no reservation");
    }
    for(std::size_t i = 0; i< blocksNum; ++i)
    {
        if(memory[i] == nullptr)
        {
            request.allocateOneBlock(*this,i);
            slot = i;
            break;
        }
    }
    if(slot == blocksNum)
    {
        for(std::size_t i = 0; i<blocksNum; ++i)
        {
            if(memory[i] != nullptr && memory[i]->getRefCount() == 0 )
            {
                delete memory[i];
                memory[i] = nullptr;
                request.allocateOneBlock(*this,i);
                slot = i;
                break;
            }
        }
    }
    if (slot == blocksNum)
    {
        throw std::logic_error("no available slots");
    }
    for(auto& r : reservations)
    {
        if(r.first == &request)
        {
            r.second--;
            reserved--;
            break;
        }
    }
    return slot;
}
void GPU::addMemoryBlock(const GenerativeRequest& request, std::size_t slot)
{
    memory[slot] = new KVBlock(blockSize);
}

void GPU::addMemoryBlock(const EmbeddingRequest& request, std::size_t slot)
{
    memory[slot] = new EmbeddingWorkspace(blockSize);
}
void GPU::assignPrefixToBlock(std::size_t block,const std::vector<std::string>& prefixTokens, const std::string& modelId)
{
    for (std::size_t i = 0; i < blocksNum; ++i)
    {
        if (i != block && hasPrefix[i] && blockModelIds[i] == modelId && prefixKeys[i] == prefixTokens)
        {
            return;
        }
    }
    blockModelIds[block] = modelId;
    prefixKeys[block] = prefixTokens;
    hasPrefix[block] = true;
}
PrefixMatch GPU::findLongestCachedPrefix(const std::string& modelId,std::size_t embeddingSize, const std::vector<std::string>& promptTokens) const
{
    PrefixMatch result{0, {}};
    std::size_t tokensPerBlock = blockSize / embeddingSize;
    for (std::size_t end = tokensPerBlock; end <= promptTokens.size();end += tokensPerBlock)
    {
        std::vector<std::string> prefix(promptTokens.begin(),promptTokens.begin() + end);
        bool found = false;
        for (std::size_t i = 0; i < blocksNum; ++i)
        {
            if (memory[i] != nullptr && hasPrefix[i] && blockModelIds[i] == modelId && prefixKeys[i] == prefix)
            {
                result.blockIds.push_back(i);
                result.matchedTokens = end;
                found = true;
                break;
            }
        }
        if (!found)
        {
            break;
        }
    }
    return result;
}

