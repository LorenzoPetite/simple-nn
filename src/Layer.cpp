#include "Net.h"

Layer::Layer(unsigned output_size)
{
	m_outputs.resize(output_size);
}

InputLayer::InputLayer(unsigned curr_layer_sz, unsigned next_layer_sz)
{
	m_weights.resize(curr_layer_sz, next_layer_sz);
}

HiddenLayer::HiddenLayer(unsigned curr_layer_sz, unsigned next_layer_sz)
{
	m_weights.resize(curr_layer_sz, next_layer_sz);
	m_z.resize(curr_layer_sz);
}

OutputLayer::OutputLayer(unsigned curr_layer_sz)
{
	m_z.resize(curr_layer_sz);
}

