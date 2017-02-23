// NeuralNetwork.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

// COMPILE : g++ -o <output-file-name> <program-name>.cpp
// RUN     : <output-file-name>


#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
using namespace std;



class TrainingData
{
public:
	TrainingData(const string filename);
	bool isEof(void) { return trainingDatafile.eof(); }
	void get_topology(vector<unsigned> &network_topology);

	unsigned get_next_input(vector<double> &input);
	unsigned get_expected_output(vector<double> &expected_output);
	
private:
	ifstream trainingDatafile;

};

TrainingData::TrainingData(const string filename)
{
	trainingDatafile.open(filename.c_str());
}

void TrainingData::get_topology(vector<unsigned> &network_topology)
{
	string line;
	string label;
	getline(trainingDatafile, line);
	stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("topology:") != 0) abort();

	while (!ss.eof())
	{
		unsigned n;
		ss >> n;
		network_topology.push_back(n);
	}
	return;
}

unsigned TrainingData::get_next_input(vector<double> &input)
{
	input.clear();
	string line;
	string label;
	getline(trainingDatafile, line);
	stringstream ss(line);
	ss >> label;
	if (label.compare("in:") == 0)
	{
		double one_value;
		while (ss >> one_value)
		{
			input.push_back(one_value);

		}
	}
	return input.size();
}

unsigned TrainingData::get_expected_output(vector<double> &expected_output)
{
	expected_output.clear();
	string line;
	string label;
	getline(trainingDatafile, line);
	stringstream ss(line);
	ss >> label;
	if (label.compare("out:") == 0)
	{
		double one_value;
		while (ss >> one_value)
		{
			expected_output.push_back(one_value);

		}
	}
	return expected_output.size();
}


class Connection
{
	public :
		double weight;
		double delta_weight;
};


// ############### NEURON / PRECEPTRON (BEGIN) ###############

class Neuron;
typedef vector<Neuron> Layer;

class Neuron
{
	public:

		Neuron									(unsigned num_outputs, unsigned _index);
		void   set_output						(double _output) { this->output = _output; };
		double get_output						(void) const     { return this->output;    };
		void   feedforward						(const Layer &prevLayer );
		void   calculate_output_layer_gradients (double expected_output );
		void   calculate_hidden_layer_gradients (const Layer &next_layer);
		void   update_input_weights				(Layer &previous_layer  );

	private:
		static double step               (double x) { return x <= 0 ? 0.0 : 1.0; }
		static double sigmoid            (double x) { return (1 / 1 + exp(-x));  }
		static double absolute_sigmoid   (double x) { return (1 / 1 + fabs(-x)); }
		static double hyperbolic_tangent (double x) { return tanh(x);            }

		static double step_derivative				(double x) { return 0.0; /*dirac delta unsupported*/}
		static double sigmoid_derivative            (double x) { return (sigmoid(x)*(1 - sigmoid(x)));  }
		static double absolute_sigmoid_derivative   (int x)    { return (1/((x+1)^2));                  }
		static double hyperbolic_tangent_derivative (int x)    { return (1-(x^2));                      }

		static double randomWeight(void)	 { return rand() / double(RAND_MAX); }
		double sumDOW (const Layer &next_layer) const;


		double		  output;
		double		  gradient;
		static double learning_rate;
		static double momentum;
		unsigned	  index;

		vector <Connection> output_weights;
};


// ---------------- Constructors / Destructors -------------------

double Neuron::learning_rate = 0.15;
double Neuron::momentum      = 0.15;

Neuron::Neuron(unsigned num_outputs, unsigned _index)
{
	// intialize vector of connections with random weights
	for (unsigned connection = 0; connection < num_outputs; ++connection)
	{
		output_weights.push_back(Connection());
		output_weights.back().weight = randomWeight();
	}
	this->index = _index;
}

// ---------------- Overloaded Operators -------------------


// ---------------- Essential Functions -------------------

void Neuron::feedforward(const Layer &previous_Layer)
{
	double weighted_sum = 0.0;
	// get the weighted sum of the outputs of previous layer
	for (unsigned neuron = 0; neuron < previous_Layer.size(); ++neuron)
	{
		weighted_sum = weighted_sum + ( previous_Layer[neuron].get_output() * previous_Layer[neuron].output_weights[index].weight );
	}

	// activation function : sigmoid
	output = hyperbolic_tangent(weighted_sum);
}


// ---------------- Helper Functions -------------------

void Neuron::calculate_output_layer_gradients(double expected_output)
{
	double delta = expected_output - output;
	gradient = delta * hyperbolic_tangent_derivative(output);

}

void Neuron::calculate_hidden_layer_gradients(const Layer &next_layer)
{
	double dow = sumDOW(next_layer);
	gradient = dow * hyperbolic_tangent_derivative(output);
}

void Neuron::update_input_weights(Layer &previous_layer)
{
	for (unsigned neuron = 0; neuron < previous_layer.size(); ++neuron)
	{
		Neuron &_neuron = previous_layer[neuron]; // easier to read

		double old_delta_weight = _neuron.output_weights[index].delta_weight;
		double new_delta_weight = learning_rate *_neuron.get_output() * gradient + momentum * old_delta_weight;

		_neuron.output_weights[index].delta_weight = new_delta_weight;
		_neuron.output_weights[index].weight = _neuron.output_weights[index].weight + new_delta_weight;


	}
}


double Neuron::sumDOW(const Layer &next_layer) const
{
	double sum = 0.0;

	// Sum the contributions to error from all the neurons
	// in the to the right of the current layer
	
	for (unsigned neuron = 0; neuron < next_layer.size()-1; ++neuron)
	{
		sum = sum + output_weights[neuron].weight * next_layer[neuron].gradient;
	}
	return sum;
}

// ############### NEURAL NETWORK ###############


class NeuralNetwork
{
	public:
		
		NeuralNetwork		(const vector<unsigned> &network_topology);
		void feedforward	(const vector<double>   &input);
		void backpropogate	(const vector<double>   &expected_output);
		void get_output		(vector<double>			&predicted_output) const;
		



	private:
		vector<Layer> network_layer; // network_layer[layer_num][neuron_num]
		double error;

};

// ---------------- Constructors / Destructors -------------------

NeuralNetwork::NeuralNetwork(const vector<unsigned> &network_topology)
{
	unsigned num_layers = network_topology.size();

	// Create Layers
	for (unsigned layer_num = 0; layer_num < num_layers; ++layer_num)
	{
		network_layer.push_back(Layer());
		unsigned num_outputs;

		// check if this is output layer
		if (layer_num == network_topology.size() - 1)	num_outputs = 0;
		// otherwise its a hidden layer
		else											num_outputs = network_topology[layer_num + 1];


		// Add bias neuron to each layer
		for (unsigned neuron_num = 0; neuron_num <= network_topology[layer_num]; ++neuron_num)
		{
			network_layer.back().push_back(Neuron(num_outputs, neuron_num));
			//cout << "Made a neuron!!" << endl;
		}

		// Set bias neuron
		network_layer.back().back().set_output(1.0);
	}
}

// ---------------- Overloaded Operators -------------------


// ---------------- Essential Functions -------------------

void NeuralNetwork::feedforward(const vector<double> &input)
{
	// check if the input size is valid - 1 for bias
	assert(input.size() == network_layer[0].size() - 1);

	// Connect values to neurons
	for (unsigned i = 0; i < input.size(); ++i)
	{
		network_layer[0][i].set_output(input[i]);
	}

	// Forward propogate

	// for each layer
	for (unsigned layer_num = 1; layer_num < network_layer.size(); ++layer_num){
		// for each neuron in current layer
		Layer &previous_layer = network_layer[layer_num - 1];
		for (unsigned neuron_num = 0; neuron_num < network_layer[layer_num].size() - 1; ++neuron_num){
		
			network_layer[layer_num][neuron_num].feedforward(previous_layer);
		}
	}
}

void NeuralNetwork::backpropogate(const vector<double> &expected_output)
{
	// Calculate global error : mean squared error

	Layer &output_layer = network_layer.back();
	error = 0.0;

	for (unsigned neuron = 0; neuron < output_layer.size() - 1; ++neuron)
	{
		// Get euclidean distance
		double delta = expected_output[neuron] - output_layer[neuron].get_output();
		// square difference and sum over all errors
		error = error + (delta*delta);
	}
	// Normalize
	error = error / (2*(output_layer.size() - 1));
	// error = sqrt(error);



	// Calculate Output Layer gradient

	for (unsigned neuron = 0; neuron < output_layer.size() - 1; ++neuron)
	{
		output_layer[neuron].calculate_output_layer_gradients(expected_output[neuron]);
	}


	// Calulate Hidden Layer gradients

	for (unsigned layer_num = network_layer.size() - 2; layer_num > 0; --layer_num)
	{
		Layer &hidden_layer = network_layer[layer_num];     // easier to read
		Layer &next_layer   = network_layer[layer_num + 1];

		for (unsigned neuron = 0; neuron < hidden_layer.size(); ++neuron)
		{
			hidden_layer[neuron].calculate_hidden_layer_gradients(next_layer);
		}

	}

	// Update connection weights

	for (unsigned layer_num = network_layer.size() - 1; layer_num > 0; --layer_num)
	{
		Layer &current_layer  = network_layer[layer_num];     // easier to read
		Layer &previous_layer = network_layer[layer_num - 1];
		
		for (unsigned neuron = 0; neuron < current_layer.size() - 1; ++neuron)
		{
			current_layer[neuron].update_input_weights(previous_layer);
		}
	}


}

void NeuralNetwork::get_output(vector<double> &predicted_output) const
{
	predicted_output.clear();
	for (unsigned neuron = 0; neuron < network_layer.back().size() - 1; ++neuron)
	{
		predicted_output.push_back(network_layer.back()[neuron].get_output());
	}
}

// ---------------- Helper Functions -------------------



// ############### MAIN ###############
void display_vector(string label , vector <double> &v)
{
	cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i)
	{
		cout << v[i] << " ";
	}
	cout << endl;
}


int main()
{

	TrainingData training_data("C:\\Users\\tiwathia\\Documents\\GitHub\\NeuralNetwork_cpp\\Training Data\\XOR_training.txt");
	vector <unsigned> network_topology;
	training_data.get_topology(network_topology);
	// e.g [3,2,1] creates 3 input neurons
	//					   2 hidden neurons
	//					   1 output neuron

	/*network_topology.push_back(3);
	network_topology.push_back(2);
	network_topology.push_back(1);*/
	NeuralNetwork myNetwork(network_topology);


	vector<double> input;
	vector<double> expected_output;
	vector<double> predicted_output;

	int training_pass = 0;
	while (!training_data.isEof())
	{
		++training_pass;
		cout << endl << "----------------------------- " << endl;
		cout << endl << "Pass " << training_pass;
		cout << endl << "----------------------------- " << endl << endl;

		// Get new input data and feed it forward
		if (training_data.get_next_input(input) != network_topology[0]) break;
		display_vector("Inputs           : ", input);
		myNetwork.feedforward(input);
		
		// Collect and display results
		myNetwork.get_output(predicted_output);
		display_vector("Predicted Output : ", predicted_output);
		
		// Train it
		training_data.get_expected_output(expected_output);
		display_vector("Expected Output  : ", expected_output);

		assert(expected_output.size() == network_topology.back());
		myNetwork.backpropogate(expected_output);

		//cout << "Network recent average error : " << myNetwork.get_recent_average_error() << endl;
		if (training_pass % 4 == 0)
		{
			cin.get();
			system("cls");
		}
	}
	
	cout << endl << "Trained." << endl;
	cin.get();
	return 0;
}




