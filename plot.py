import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_json_values(directory):
    """
    Reads JSON files in the specified directory, extracts x and y values,
    and plots them for each unique k value. Saves each plot as a PNG file.
    
    Args:
        directory (str): Path to the directory containing JSON files.
    """
    # Dictionary to store x and y values for each k
    data = {}
    temp_list = []
    field = "acc"
    label = "Accuracy"
    layer = 4
    # Iterate through all files in the directory
    print("Number of files: ", len(os.listdir(directory)))
    for filename in np.sort(os.listdir(directory)):
        # Check if the file follows the format k_x_y.json
        if filename.endswith(".json") and "_" in filename:
            try:
                # Extract k, x, and y from the filename
                parts = filename.split("_")[-3:]
                k = int(parts[0])  # k value
                if k==layer:
                    if k not in data.keys():
                        print(k)
                        data[k] = []
                    else:
                        pass
                    
                    recent = float(parts[1])
                    heavy = float(parts[2].replace(".json", ""))
                    print(recent, heavy)

                    # Read JSON file
                    filepath = os.path.join(directory, filename)
                    with open(filepath, "r") as file:
                        json_data = json.load(file)
                        temp_list.append(float(json_data[field]))
                    if len(temp_list)==9:
                        tempo = temp_list.copy()
                        data[k].append(tempo)
                        temp_list = []               
                # # Loop through each row to plot it


            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    # print(np.array(data[2]).shape)
    # # print(np.array(data[4]).shape)
    # assert 1==2
    # Plot data for each k value
    for i, row in enumerate(data[layer]):
        if i==0 or i==2 or i==7:
            x = np.array(range(len(row)))/10 # x-axis: indices of the row
            y = row              # y-axis: values in the row
            plt.plot(x, y, label=f'Recent budget ratio: {i/10}')  # Add a label for each line

        # Add labels and legend
    plt.axhline(y=0.7554, color='red', linestyle='--', label='Full KV Cache')
    plt.axhline(y=0.7248, color='black', linestyle='--', label='FastV without Heavy hitters')
    plt.xlabel('Heavy budget ratio')
    plt.ylabel(label)
    plt.title(f'Heavy budget vs {label} for a fixed recent budget')
    # plt.ylim(0, 2)
    plt.legend()
    output_file = os.path.join( f"plot_{label}_k_{layer}.png")
    plt.savefig(output_file)

    # # Show the plot
    # plt.show()
    # for k, values in data.items():
    #     plt.figure()
    #     plt.plot(values["x"], values["y"], marker="o", label=f"k={k}")
    #     plt.title(f"Plot for k={k}")
    #     plt.xlabel("x-axis")
    #     plt.ylabel("y-axis")
    #     plt.legend()
    #     plt.grid(True)

    #     # Save plot as PNG file
    #     output_file = os.path.join(directory, f"plot_k_{k}.png")
    #     plt.savefig(output_file)
    #     print(f"Saved plot for k={k} as {output_file}")

    #     # Close the plot to free memory
    #     plt.close()

# Example usage
directory_path = "aokvqa_eval_fastv"  # Replace with your directory path
plot_json_values(directory_path)
