#include "Request.h"

Request::Request(int id, const Model& model, int admissionCycle, std::string prompt):
    id(id), model(model), admissionCycle(admissionCycle), tokens(), memoryBlocks(), reservedBlocks(0)
{
    // Tokenization logic would go here
}

Request::~Request() {
    // Destructor logic if needed
}

void Request::deallocateMemory(GPU& gpu) {
    for (std::size_t blockIndex : memoryBlocks) {
        gpu.deallocateMemoryBlock(blockIndex);
    }
    memoryBlocks.clear();
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