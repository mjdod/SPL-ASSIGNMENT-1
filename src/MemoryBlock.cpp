#include "MemoryBlock.h"

MemoryBlock::MemoryBlock(size_t size): size(size), refCount(0) {
    data = new int[size];
    for (size_t i = 0; i < size; ++i) {
        data[i] = 0;
    }
}
MemoryBlock::readFromBlock(std::size_t offset) const {
    if (offset >= size) {
        throw std::out_of_range("Offset is out of bounds");
    }
    return data[offset];
}
MemoryBlock::writeToBlock(std::size_t offset, int value) {
    if (offset >= size) {
        throw std::out_of_range("Offset is out of bounds");
    }
    data[offset] = value;
}
MemoryBlock::acquire() {
    refCount++;
}
MemoryBlock::release() {
    refCount--;
    if (refCount == 0) {
        return onLastRelease();
    }
    return IdleAction::KeepCached;
}
// Destructor
MemoryBlock::~MemoryBlock() {
    delete[] data;
}
//copy constructor
MemoryBlock::MemoryBlock(const MemoryBlock& other): size(other.size), refCount(other.refCount) {
    delete[] data;
    data = new int[size];
    for (size_t i = 0; i < size; ++i) {
        data[i] = other.data[i];
    }
}
//copy assignment
MemoryBlock::MemoryBlock(MemoryBlock&& other) noexcept: size(other.size), refCount(other.refCount), data(other.data) {
    other.data = nullptr;
}
//move constructor
MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other) {
    if (this != &other) {
        delete[] data;
        size = other.size;
        refCount = other.refCount;
        data = new int[size];
        for (size_t i = 0; i < size; ++i) {
            data[i] = other.data[i];
        }
    }
    return *this;
}
//move assignment
MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other) noexcept {
    if (this != &other) {
        delete[] data;
        size = other.size;
        refCount = other.refCount;
        data = other.data;
        other.data = nullptr;
    }
    return *this;
}

KVBlock::KVBlock(size_t size): MemoryBlock(size) {}
IdleAction KVBlock::onLastRelease() const {
    return IdleAction::KeepCached;
}
MemoryBlock* KVBlock::clone() const {
    return new KVBlock(*this);
}

EmbeddingWorkspace::EmbeddingWorkspace(size_t size): MemoryBlock(size) {}
IdleAction EmbeddingWorkspace::onLastRelease() const {
    return IdleAction::Destroy;
}
MemoryBlock* EmbeddingWorkspace::clone() const {
    return new EmbeddingWorkspace(*this);
}