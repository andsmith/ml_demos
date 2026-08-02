import matplotlib.pyplot as plt
import numpy as np
import time

# Enable interactive mode
#plt.ion()

# Create initial data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create the figure and axes
fig, ax = plt.subplots()
poly_collection = ax.stackplot(x, y1, y2)
plt.show()
# Main loop to update the plot
for i in range(100):
    # Update the data
    y1 = np.sin(x + i * 0.1)
    y2 = np.cos(x + i * 0.1)

    # Update the stackplot
    v1 = np.stack([x, y1], axis=-1)
    v2 = np.stack([x, y1 + y2], axis=-1)


    poly_collection[0].set_verts(v1)
    poly_collection[1].set_verts(v2)

    # Redraw the figure
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.1)

# Keep the window open after the animation
plt.waitforbuttonpress()