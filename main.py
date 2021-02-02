

# import numpy as np
# from lets_plot import *
#
# # Generate random data-points for the demo.
# np.random.seed(12)
# data = dict(
#     cond=np.repeat(['A', 'B'], 200),
#     rating=np.concatenate((np.random.normal(0, 1, 200), np.random.normal(1, 1.5, 200)))
# )
#
# # Create plot specification.
# p = ggplot(data, aes(x='rating', fill='cond')) + ggsize(500, 250) \
#     + geom_density(color='dark_green', alpha=.7) + scale_fill_brewer(type='seq') \
#     + theme(axis_line_y='blank')
#
# # Display plot in 'SciView'.
# p.show()


import matplotlib.pyplot as plt
from matplotlib.text import Text

plt.title ("Teste - 000")
Text(0.5, 1.0, 'Teste')
plt.show()


# from bokeh.plotting import figure, output_notebook, show
#
#
# import numpy as np
# # Generate random data
# x = np.arange(1, 11)
# y = np.random.rand(10)
#
#
# # Generate canvas
# fig = figure(title='Line Chart Example',
#              x_axis_label='x',
#              y_axis_label='y',
#              width=800,
#              height=400)
#
# # Draw the line
# fig.line(x, y,
#          line_alpha=0.8,
#          legend_label='example value',
#          line_width=2)
#
# show(fig)
#
#






