#include <iostream>
#include <GPU.h>
#include <MemoryBlock.h>
#include <Request.h>
//constructer (still need to implment rule of 5)
GPU::GPU(std::size_t blocksNum, std::size_t blockSize): blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum,nullptr), reserved(0)
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
