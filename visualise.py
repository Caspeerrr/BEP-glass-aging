import matplotlib.pyplot as plt


def visualise(timesteps, Data): 
    """
    visualises the data found
    """

    for key, value in Data:

        plt.scatter(timesteps, value)
        plt.xlabel('timesteps')
        plt.title(key)
        plt.show()