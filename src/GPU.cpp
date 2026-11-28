#include <iostream>
#include <GPU.h>
//constructer (still need to implment rule of 5)
GPU::GPU(std::size_t blocksNum, std::size_t blockSize): blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum,nullptr)
{
}



//the functions required in implementation O-O
std::size_t GPU::availableSlots() const
{

}
