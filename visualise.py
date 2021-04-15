import matplotlib.pyplot as plt


def visualise(timesteps, msd, vsd, mnn_distance, vnn_distance):
    """
    visualises the data found
    msd: mean squared displacement
    vsd: variance square displacement
    mnn_distance: mean nearest neighbour distance
    vnn_distance: variance nearest neighbour distance
    """

    plt.scatter(timesteps, msd)
    plt.xlabel('timesteps')
    plt.ylabel('mean')
    plt.title('mean square displacement')
    plt.show()

    plt.scatter(timesteps, vsd)
    plt.xlabel('timesteps')
    plt.title('variance square displacement')
    plt.show()

    plt.scatter(timesteps, mnn_distance)
    plt.xlabel('timesteps')
    plt.ylabel('mean')
    plt.title('mean nn distance')
    plt.show()

    plt.scatter(timesteps, vnn_distance)
    plt.xlabel('timesteps')
    plt.ylabel('mean')
    plt.title('variance nn distance')
    plt.show()
