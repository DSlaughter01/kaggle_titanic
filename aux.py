# https://www.researchgate.net/publication/221996720_Alternatives_to_Median_Absolute_Deviation
# Use Rousseeuw and Croux's 1993 paper to identify outliers using the median 
# Their approach is more robust than the Median Absolute Deviation (MAD) which is nonetheless simply calculated and useful on symmetric data
"""
def handle_outliers(data, col, z = 2.5, plot = True):

    data[col] = list(data[col])
    if 0.0 in data[col]: print(True)
    # Transform the data by taking a logarithm
    transformed_col = np.log(data[col])
    print(transformed_col)

    # Calculate its mean and standard deviation
    mean = np.mean(transformed_col)
    sd = np.std(transformed_col)

    # Currently shows -inf and nan LOOOOL
    # print(mean)
    # print(sd)

    # Upper and lower quartiles as a value to change outliers to, as a sort of compression
    q1 = np.quantile(transformed_col, 0.25)
    q3 = np.quantile(transformed_col, 0.75)

    # Modify outliers - this is my idea and not based on any sources, but involves changing outliers to the value of q3 or q1 as appropriate
    for cell in transformed_col:
        if cell > mean + z * sd: cell = q3
        elif cell < mean - z * sd: cell = q1

    # Plot if required
    if plot:

        fig = plt.figure()
        x = data["PassengerId"]
        y = transformed_col

        plt.plot(x, y, ".")
        plt.title(f"{col}")
        plt.xlabel("Passenger ID")
        plt.ylabel(f"{col} Distribution")
        # plt.show()

    return transformed_col

"""