// COMPILE : g++ -o <output-file-name> <program-name>.cpp
// RUN     : <output-file-name>


#include <iostream>
#include <vector>

using namespace std;


// ############### NEURON / PRECEPTRON (BEGIN)###############

class Neuron
{
	public:

	private:
};

// ---------------- Constructors / Destructors -------------------


// ---------------- Overloaded Operators -------------------


// ---------------- Essential Functions -------------------


// ---------------- Helper Functions -------------------


// ############### NEURAL NETWORK ###############

typedef vector<Neuron> Layer;
class NeuralNetwork
{
	public:
		NeuralNetwork      (const vector<unsigned> &network_topology);
		void feedforward   (const vector<double> &input) {};
		void backpropogate (const vector<double> &expected_output) {};
		void get_output    (vector<double> predicted_output) const {};

	private:
		vector<Layer> network_layer; // network_layer[layer_num][neuron_num]

};

// ---------------- Constructors / Destructors -------------------

NeuralNetwork::NeuralNetwork(const vector<unsigned> &network_topology)
{
	unsigned num_layers = network_topology.size();

	// Create Layers
	for(unsigned layer_num = 0; layer_num < num_layers; ++layer_num)
	{
		network_layer.push_back(Layer());

		// Add bias neuron to each layer
		for(unsigned neuron_num = 0; neuron_num <= network_topology[layer_num]; ++neuron_num)
		{
			network_layer.back().push_back(Neuron());
			cout << "Made a neuron!!" << endl;
		}
	}
}

// ---------------- Overloaded Operators -------------------


// ---------------- Essential Functions -------------------


// ---------------- Helper Functions -------------------


// ############### MAIN ###############

int main()
{
	vector <unsigned> network_topology;
	// e.g [3,2,1] creates 3 input neurons
	//					   2 hidden neurons
	//					   1 output neuron

	network_topology.push_back(3);
	network_topology.push_back(2);
	network_topology.push_back(1);
	NeuralNetwork myNetwork (network_topology);


	vector<double> input;
	vector<double> expected_output;
	vector<double> predicted_output;
	myNetwork.feedforward(input);
	myNetwork.backpropogate(expected_output);
	myNetwork.get_output(predicted_output);

	return 0;
}


