#include <fstream>
#include <stdexcept>
#include "json.hpp"

#include "Server.h"

using namespace std;
using json = nlohmann::json;

Config Server::readConfig(string configPath) const{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for(auto& gpu: config["gpus"]){
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for(auto& model: config["models"]){
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for(auto& request: config["requests"]){
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if(typeStr != "generative" && typeStr != "text_embedder"){
            throw invalid_argument("invalid request type");
        }
        RequestType type = typeStr=="generative"?RequestType::Generative:RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0) {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0) {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id, type, modelId, admissionCycle, prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0) {
        throw invalid_argument("missing scheduling policy");
    }
    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}

Server::Server(string configPath): config(readConfig(configPath)) {
    for (const auto& gpuConfig: config.gpuConfigs) {
        gpus.push_back(new GPU(gpuConfig.memoryBytes/config.blockSize, config.blockSize));
    }
    for (const auto& modelConfig: config.modelConfigs) {
        models[modelConfig.id] = modelConfig;
    }
}
Server::step() {
    // Implement the logic to process requests based on the scheduling policy.
    // This may involve checking the current state of requests, moving them between
    // non-submitted, waiting, and active states, and ensuring that GPU memory
    // constraints are respected.

    // Example pseudocode:
    // 1. Check for new requests to submit.
    // 2. Move eligible requests from non-submitted to waiting.
    // 3. Schedule waiting requests to active based on the policy.
    // 4. Update GPU memory usage and check for completion of active requests.
    // 5. Return true if any changes were made, false otherwise.

    return true; // Placeholder return value; replace with actual logic.
}
Server::serve(){
    while (true) {
        bool changed = step();
        if (!changed) {
            // If no changes were made, we can break the loop or wait for new requests.
            break;
        }
    }
}

Server::~Server() {
    for (auto gpu: gpus) {
        delete gpu;
    }
    for (auto model: models) {
        delete model.second;
    }
    for (auto request: nonSubmittedRequests) {
        delete request;
    }
    for (auto request: waitingRequests) {
        delete request;
    }
    for (auto request: activeRequests) {
        delete request;
    }
    delete policy;
}
Server::Coppy(const Server& other) {
    config = other.config;
    for (const auto& gpu: other.gpus) {
        gpus.push_back(new GPU(*gpu));
    }
    for (const auto& modelPair: other.models) {
        models[modelPair.first] = new ModelConfig(*modelPair.second);
    }
    for (const auto& request: other.nonSubmittedRequests) {
        nonSubmittedRequests.push_back(new RequestConfig(*request));
    }
    for (const auto& request: other.waitingRequests) {
        waitingRequests.push_back(new RequestConfig(*request));
    }
    for (const auto& request: other.activeRequests) {
        activeRequests.push_back(new RequestConfig(*request));
    }
    policy = new SchedulingPolicy(*other.policy);
}
Server::CopyAssignment(const Server& other) {
    if (this != &other) {
        for (auto gpu: gpus) {
            delete gpu;
        }
        for (auto model: models) {
            delete model.second;
        }
        for (auto request: nonSubmittedRequests) {
            delete request;
        }
        for (auto request: waitingRequests) {
            delete request;
        }
        for (auto request: activeRequests) {
            delete request;
        }
        delete policy;

        Coppy(other);
    }
    return *this;
}
Server::Move(Server&& other) noexcept {
    config = std::move(other.config);
    gpus = std::move(other.gpus);
    models = std::move(other.models);
    nonSubmittedRequests = std::move(other.nonSubmittedRequests);
    waitingRequests = std::move(other.waitingRequests);
    activeRequests = std::move(other.activeRequests);
    policy = other.policy;

    other.policy = nullptr;
}
Server::MoveAssignment(Server&& other) noexcept {
    if (this != &other) {
        for (auto gpu: gpus) {
            delete gpu;
        }
        for (auto model: models) {
            delete model.second;
        }
        for (auto request: nonSubmittedRequests) {
            delete request;
        }
        for (auto request: waitingRequests) {
            delete request;
        }
        for (auto request: activeRequests) {
            delete request;
        }
        delete policy;
    }
    Move(std::move(other));
    return *this;
}