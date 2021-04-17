import matplotlib.pyplot as plt


def visualise(timesteps, Data): 
    """
    visualises the data found
    """

    for key, value in Data.items():

        plt.scatter(timesteps, value)
        plt.xlabel('timesteps')
        plt.title(key)
        plt.show()