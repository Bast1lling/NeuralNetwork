#include <nn/layer/affine.hpp>
#include <nn/layer/relu.hpp>
#include <nn/network/network.hpp>
#include <ranges>

namespace network {
    Network::Network(size_t batch_size, size_t input_size, size_t output_size) : weights(0){
        model.reserve(2);
        model = {
            std::make_unique<layer::Layer>(layer::Affine(batch_size, input_size, output_size)),
            std::make_unique<layer::Layer>(layer::ReLu(batch_size, output_size))
        };
        weights.reserve(model.size());
        for (const auto layer: model) {
            weights.emplace_back(layer->weights());
        }
    }

    const math::Matrix &Network::forward(const math::Matrix &in) {
        auto input = in;
        for (const std::unique_ptr<layer::Layer> layer: model) {
            input = layer->forward(input);
        }
        return input;
    }

    std::vector<const math::Matrix &> Network::backward(math::Matrix &&dloss) {
        std::vector<const math::Matrix &> result = {dloss};
        for (auto & it : std::ranges::reverse_view(model)) {
            result.emplace_back(it->backward(result[result.size() - 1]));
        }
        return result;
    }

    float Network::train(const math::Matrix &X, const math::Matrix &y) {
        const math::Matrix &y_out = forward(X);
        auto [fst, snd] = loss_function(y_out, y);
        const auto grads = backward(std::move(snd));
        optimizer(grads, weights, optimizer::OptimizerParameters());
        return fst;
    }
} // namespace network
