#imports
import matplotlib.pyplot as plt
import numpy as np


def preprocess_dictionary(input_dictionary):
    """
        Preprocess the input dictionary so it can be plotted
        Args:
            input_dictionary (dict): input dictionary of {concept:{layer:{class: tcav value}}}

        Returns:
            plot_dict (dict) : output dictionary of {class:{layer:[tcav values]}}, tcav values ordered by input concepts
            labels (list): List of concept names
        """
    labels = []
    plot_dict = {}

    for concept, concept_values in input_dictionary.items():
        labels.append(concept)
        for layer, layer_values in concept_values.items():
            for classes, classes_score in layer_values.items():

                if classes not in plot_dict.keys():
                    plot_dict[classes] = {layer: [round(classes_score, 2)]}

                elif layer not in plot_dict[classes].keys():
                    plot_dict[classes][layer] = [round(classes_score, 2)]

                else:
                     plot_dict[classes][layer].append(round(classes_score, 2))
    return plot_dict, labels

def plot_class_tcav_scores(class_dict, labels, class_name):
    """
        Plot the input class dictionary, save at result_plots folder
        Args:
            class_dict (dict): input dictionary of {layer:[tcav values]}, tcav values ordered the same as is labels ordering
            labels (list): List of concept names
            class_name (str): class name

    """
    
    layer_1_list = class_dict["layer1"]
    layer_2_list = class_dict["layer2"]
    layer_3_list = class_dict["layer3"]
    layer_4_list = class_dict["layer4"]
    layer_5_list = class_dict["layer5"]
    layer_6_list = class_dict["layer6"]
    
    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width*2.5, layer_1_list, width, label='layer1')
    rects2 = ax.bar(x - width*1.25, layer_2_list, width, label='layer2')
    rects3 = ax.bar(x, layer_3_list, width, label='layer3')
    rects4 = ax.bar(x + width*1.25, layer_4_list, width, label='layer4')
    rects5 = ax.bar(x + width*2.5, layer_5_list, width, label='layer5')
    rects6 = ax.bar(x + width*3.75, layer_6_list, width, label='layer5')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title(f'Tcav scores for the class {class_name}')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    ax.bar_label(rects5, padding=3)
    ax.bar_label(rects6, padding=3)

    fig.tight_layout()
    plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0))
    fig.savefig(f"result_plots\\tcav_scores_{class_name}", bbox_inches='tight', pad_inches=0.5)
    plt.close()


def make_plots(tcav_dict):
    """
    Main plotting function that creates plots to compare the tcav scores accross concepts and layers for each class
    Args:
        input_dictionary (dict): input dictionary of {concept:{layer:{class: tcav value}}}

    """
    plot_dict, labels = preprocess_dictionary(tcav_dict)
    for class_name, class_dict in plot_dict.items():
        plot_class_tcav_scores(class_dict, labels, class_name)

if __name__ == "__main__":
    pass