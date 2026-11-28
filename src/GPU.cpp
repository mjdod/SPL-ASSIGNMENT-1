#include <iostream>
#include "GPU.h"
#include "Request.h"
#include "MemoryBlock.h"
#include <stdexcept>
//constructer and rule of 5
GPU::GPU(std::size_t blocksNum, std::size_t blockSize): blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum,nullptr), reserved(0),reservations(),prefixKeys(blocksNum),blockModelIds(blocksNum),hasPrefix(blocksNum, false),evicted(0)
{
}
//Destructor
GPU::~GPU()
{
    for (MemoryBlock* block : memory)
    {
        delete block;
    }
}
//copy constructor
GPU::GPU(const GPU& other): blocksNum(other.blocksNum),blockSize(other.blockSize),memory(other.blocksNum, nullptr),reserved(other.reserved),reservations(other.reservations),prefixKeys(other.prefixKeys),blockModelIds(other.blockModelIds),hasPrefix(other.hasPrefix),evicted(other.evicted)
{
    try
    {
        for (std::size_t i = 0; i < blocksNum; ++i)
        {
            if (other.memory[i] != nullptr)
            {
                memory[i] = other.memory[i]->clone();
                std::size_t count = other.memory[i]->getRefCount();

                for (std::size_t j = 0; j < count; ++j)
                {
                    memory[i]->acquire();
                }
            }
        }
    }
    catch (...)
    {
        for (MemoryBlock* block : memory)
        {
            delete block;
        }
        throw;
    }
}

//copy assignment
GPU& GPU::operator=(const GPU& other)
{
    if (this != &other)
    {
        GPU copy(other);
        *this = std::move(copy);
    }
    return *this;
}
//move constructor
GPU::GPU(GPU&& other) noexcept: blocksNum(other.blocksNum),blockSize(other.blockSize),memory(std::move(other.memory)),reserved(other.reserved),reservations(std::move(other.reservations)),prefixKeys(std::move(other.prefixKeys)),blockModelIds(std::move(other.blockModelIds)),hasPrefix(std::move(other.hasPrefix)),evicted(other.evicted)
{
    other.blocksNum = 0;
    other.blockSize = 0;
    other.reserved = 0;
    other.evicted = 0;
}
//move assignment
GPU& GPU::operator=(GPU&& other) noexcept
{
    if (this != &other)
    {
        for (MemoryBlock* block : memory)
        {
            delete block;
        }
        blocksNum = other.blocksNum;
        blockSize = other.blockSize;
        memory = std::move(other.memory);
        reserved = other.reserved;
        reservations = std::move(other.reservations);
        prefixKeys = std::move(other.prefixKeys);
        blockModelIds = std::move(other.blockModelIds);
        hasPrefix = std::move(other.hasPrefix);
        evicted = other.evicted;
        other.blocksNum = 0;
        other.blockSize = 0;
        other.reserved = 0;
        other.evicted = 0;
    }
    return *this;
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
//reservation logic and dealing with requests
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

                hasPrefix[i] = false;
                prefixKeys[i].clear();
                blockModelIds[i].clear();

                ++evicted;
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
    request.allocateOneBlock(*this, slot);
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
//prefix logic
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
void GPU::copyMemoryFromOtherGPU(const GPU& other, const std::vector<std::size_t>& blocks)
{
    if (this == &other)
    {
        return;
    }
    std::vector<bool> protectedSlots(blocksNum, false);
    for (std::size_t sourceSlot : blocks)
    {
        if (sourceSlot >= other.blocksNum || other.memory[sourceSlot] == nullptr || !other.hasPrefix[sourceSlot])
        {
            throw std::logic_error("invalid source cache block");
        }
        for (std::size_t i = 0; i < blocksNum; ++i)
        {
            if (memory[i] != nullptr &&
                hasPrefix[i] &&
                blockModelIds[i] == other.blockModelIds[sourceSlot] &&
                prefixKeys[i] == other.prefixKeys[sourceSlot])
            {
                protectedSlots[i] = true;
            }
        }
    }
    for (std::size_t sourceSlot : blocks)
    {
        bool alreadyExists = false;
        for (std::size_t i = 0; i < blocksNum; ++i)
        {
            if (memory[i] != nullptr && hasPrefix[i] && blockModelIds[i] == other.blockModelIds[sourceSlot] && prefixKeys[i] == other.prefixKeys[sourceSlot])
            {
                alreadyExists = true;
                break;
            }
        }
        if (alreadyExists)
        {
            continue;
        }
        std::size_t slot = blocksNum;
        for (std::size_t i = 0; i < blocksNum; ++i)
        {
            if (memory[i] == nullptr)
            {
                slot = i;
                break;
            }
        }
        if (slot == blocksNum)
        {
            for (std::size_t i = 0; i < blocksNum; ++i)
            {
                if (memory[i] != nullptr &&
                    memory[i]->getRefCount() == 0 &&
                    !protectedSlots[i])
                {
                    slot = i;
                    break;
                }
            }
        }
        if (slot == blocksNum)
        {
            throw std::logic_error("no available slot for copying");
        }
        MemoryBlock* copiedBlock = other.memory[sourceSlot]->clone();
        if (memory[slot] != nullptr)
        {
            delete memory[slot];
            ++evicted;
        }
        memory[slot] = copiedBlock;
        hasPrefix[slot] = true;
        prefixKeys[slot] = other.prefixKeys[sourceSlot];
        blockModelIds[slot] = other.blockModelIds[sourceSlot];

        protectedSlots[slot] = true;
    }
}

void GPU::release(std::size_t slot)
{
    if (slot >= blocksNum || memory[slot] == nullptr)
    {
        throw std::out_of_range("invalid slot");
    }
    IdleAction action = memory[slot]->release();
    if (memory[slot]->getRefCount() == 0)
    {
        if (action == IdleAction::Destroy || !hasPrefix[slot])
        {
            delete memory[slot];
            memory[slot] = nullptr;
            hasPrefix[slot] = false;
            prefixKeys[slot].clear();
            blockModelIds[slot].clear();
        }
    }
}
//reading and writing for block
void GPU::acquire(std::size_t slot)
{
    if (slot >= blocksNum || memory[slot] == nullptr)
    {
        throw std::out_of_range("invalid slot");
    }
    memory[slot]->acquire();
}
int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const
{
    if (blockId >= blocksNum || memory[blockId] == nullptr)
    {
        throw std::out_of_range("invalid block");
    }
    return memory[blockId]->readFromBlock(offset);
}

void GPU::writeToBlock(std::size_t blockId,std::size_t offset, int value)
{
    if (blockId >= blocksNum || memory[blockId] == nullptr)
    {
        throw std::out_of_range("invalid block");
    }
    memory[blockId]->writeToBlock(offset, value);
}