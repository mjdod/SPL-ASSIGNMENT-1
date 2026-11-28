#include "Request.h"
#include "Model.h"
#include "GPU.h"

Request::Request(int id, const Model& model, int admissionCycle, std::string prompt):
    id(id), model(model), admissionCycle(admissionCycle), tokens(model.tokenizer(prompt)), memoryBlocks(), reservedBlocks(0)
{
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
//GenerativeRequest
GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt, std::string scriptedOutput,
    std::size_t maxOutputTokens)
    : Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput), maxOutputTokens(maxOutputTokens) {
}
std::size_t GenerativeRequest::getMaxNeededMemory() const{
    return maxOutputTokens;
}
std::size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const {
    return ;
}
std::vector<std::string> GenerativeRequest::getCacheableTokens() const {
    return tokens;
}
void GenerativeRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const {
    gpu.allocateBlock(*this);
}
Request* GenerativeRequest::clone(const Model& model) const {
    return new GenerativeRequest(id, model, admissionCycle,prompt, scriptedOutput, maxOutputTokens);//???
}
void GenerativeRequest::act(GPU& gpu) {
    // Implementation of the act method for GenerativeRequest
    // Add the specific logic for how a generative request should act on the GPU
}
bool GenerativeRequest::finished() const {
    // Implementation of the finished method for GenerativeRequest
    // Return true if the generative request has completed its task, false otherwise
}
void GenerativeRequest::reserveMemory(GPU& gpu) {
    // Implementation of the reserveMemory method for GenerativeRequest
    // Add the specific logic for how a generative request should reserve memory on the GPU
}
std::string GenerativeRequest::getOutput() const {
    return scriptedOutput;
}
std::size_t GenerativeRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}
