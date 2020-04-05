import numpy as np
from scipy.stats import uniform, norm, semicircular
import pandas as pd

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Panel
from bokeh.models.widgets import Slider, Tabs, Div
from bokeh.layouts import row, WidgetBox
from bokeh.plotting import figure


def make_dataset_wigner(N, gen_type, div_):
    """Creates a ColumnDataSource object with data to plot.
    """
    if gen_type == 1:
        gen = uniform(
            loc=-uniform.mean(scale=1/uniform.std()),
            scale=1/uniform.std()
        )
        gen_type_str = 'Uniform'
    else:
        gen = norm
        gen_type_str = 'Gaussian'

    assert gen.mean() == 0.0, 'Random generator should be centered.'
    assert gen.std() == 1.0, 'Random generator should be of unit variance.'

    A = np.empty((N, N))
    A[np.triu_indices(N, 1)] = gen.rvs(size=N*(N-1)//2)
    A[np.tril_indices(N, -1)] = A.T[np.tril_indices(N, -1)]
    A[np.diag_indices(N)] = gen.rvs(size=N)
    A /= np.sqrt(N)

    spectrum = np.linalg.eigvalsh(A)
    hist, edges = np.histogram(spectrum, density=True, bins=50)

    df = pd.DataFrame({'top': hist,
                       'bottom': 0,
                       'left': edges[:-1],
                       'right': edges[1:],
                       })

    xx = np.linspace(-2, 2, 200)
    pdf = semicircular.pdf(xx, scale=2)
    df_pdf = pd.DataFrame({'xx': xx, 'pdf': pdf})

    params_text = '<b>Parameters:</b><br><ul><li>Entries distribution = {:s}</li></ul>'.format(gen_type_str)
    div_.text = params_text

    # Convert dataframe to column data source
    return ColumnDataSource(df), ColumnDataSource(df_pdf)


def make_dataset_mp(N, n, gen_type, div_):
    """Creates a ColumnDataSource object with data to plot.
    """
    if gen_type == 1:
        gen = uniform(
            loc=-uniform.mean(scale=1/uniform.std()),
            scale=1/uniform.std()
        )
        gen_type_str = 'Uniform'
    else:
        gen = norm
        gen_type_str = 'Gaussian'

    assert gen.mean() == 0.0, 'Random generator should be centered.'
    assert gen.std() == 1.0, 'Random generator should be of unit variance.'

    c = N / n
    lambda_minus = (1 - np.sqrt(c)) ** 2
    lambda_plus = (1 + np.sqrt(c)) ** 2

    X = gen.rvs(size=N*n).reshape(N, n)
    A = np.dot(X, X.T)/n

    spectrum = np.linalg.eigvalsh(A)
    hist, edges = np.histogram(spectrum, density=True, bins=50)

    df = pd.DataFrame({'top': hist,
                       'bottom': 0,
                       'left': edges[:-1],
                       'right': edges[1:],
                       })

    xx = np.linspace(np.min(spectrum) * 0.9, np.max(spectrum) * 1.1, 200)
    marcenko_pastur_pdf = lambda x: np.sqrt((lambda_plus-x)*(x-lambda_minus))/(2*np.pi*c*x) if x > lambda_minus and x < lambda_plus else 0.0

    pdf = list(map(marcenko_pastur_pdf, xx))
    df_pdf = pd.DataFrame({'xx': xx, 'pdf': pdf})

    params_text = '<b>Parameters:</b><br><ul><li>Entries distribution = {:s}</li></ul>'.format(gen_type_str)
    div_.text = params_text

    # Convert dataframe to column data source
    return ColumnDataSource(df), ColumnDataSource(df_pdf)


def make_plot_wigner(src_wigner, src_pdf_wigner):
    """Create a figure object to host the plot.
    """
    # Blank plot with correct labels
    fig_wigner = figure(plot_width=700,
                        plot_height=700,
                        title="Wigner's semicircle law",
                        )

    fig_wigner.quad(top='top',
                    bottom='bottom',
                    left='left',
                    right='right',
                    source=src_wigner,
                    fill_color='navy',
                    line_color='white',
                    alpha=0.5,
                    )

    # Continuous pdf
    fig_wigner.line('xx',
                    'pdf',
                    source=src_pdf_wigner,
                    color='color',
                    legend='Semicircle distribution',
                    line_color='red',
                    line_dash='dashed',
                    )

    fig_wigner.legend.click_policy = 'hide'
    fig_wigner.legend.location = 'top_right'
    return fig_wigner


def make_plot_mp(src_mp, src_pdf_mp):
    """Create a figure object to host the plot.
    """
    # Blank plot with correct labels
    fig_mp = figure(plot_width=700,
                    plot_height=700,
                    title="Marcenko-Pastur's distribution",
                    )

    fig_mp.quad(top='top',
                bottom='bottom',
                left='left',
                right='right',
                source=src_mp,
                fill_color='navy',
                line_color='white',
                alpha=0.5,
                )

    # Continuous pdf
    fig_mp.line('xx',
                'pdf',
                source=src_pdf_mp,
                color='color',
                legend='Marcenko-Pastur',
                line_color='red',
                line_dash='dashed',
                )

    fig_mp.legend.click_policy = 'hide'
    fig_mp.legend.location = 'top_right'
    return fig_mp


def update_wigner(attr, old, new):
    """Update ColumnDataSource object.
    """
    # Change to selected values
    gen_type = gen_type_select_wigner.value
    N = N_select_wigner.value

    # Create new ColumnDataSource
    new_src_wigner, new_src_pdf_wigner = make_dataset_wigner(N, gen_type, div_wigner)

    # Update the data on the plot
    src_wigner.data.update(new_src_wigner.data)
    src_pdf_wigner.data.update(new_src_pdf_wigner.data)


def update_mp(attr, old, new):
    """Update ColumnDataSource object.
    """
    # Change to selected values
    gen_type = gen_type_select_mp.value
    N = N_select_mp.value
    n = n_select_mp.value

    # Create new ColumnDataSource
    new_src_mp, new_src_pdf_mp = make_dataset_mp(N, n, gen_type, div_wigner)

    # Update the data on the plot
    src_mp.data.update(new_src_mp.data)
    src_pdf_mp.data.update(new_src_pdf_mp.data)


######################################################################
###
### WIGNER
###
######################################################################
# Slider to select parameters
gen_type_select_wigner = Slider(start=1,
                               end=2,
                               step=1,
                               title='Entries distribution',
                               value=1,
                               )

N_select_wigner = Slider(start=1,
                        end=2000,
                        step=1,
                        value=500,
                        title='Number of samples'
                        )

# Update the plot when parameters are changed
gen_type_select_wigner.on_change('value', update_wigner)
N_select_wigner.on_change('value', update_wigner)

div_wigner = Div(text='<b>Parameters:</b><br>', width=300, height=100)

src_wigner, src_pdf_wigner = make_dataset_wigner(N_select_wigner.value, gen_type_select_wigner.value, div_wigner)

fig_wigner = make_plot_wigner(src_wigner, src_pdf_wigner)

controls_wigner = WidgetBox(N_select_wigner,
                            gen_type_select_wigner,
                            div_wigner,
                            width=300,
                            )

# Create a row layout
layout_wigner = row(controls_wigner, fig_wigner)

# Make a tab with the layout

tab_wigner = Panel(child=layout_wigner, title='Wigner')

######################################################################
###
### MARCENKO-PASTUR
###
######################################################################
# Slider to select parameters
gen_type_select_mp = Slider(start=1,
                            end=2,
                            step=1,
                            title='Entries distribution',
                            value=1,
                            )

N_select_mp = Slider(start=1,
                     end=2000,
                     step=1,
                     value=200,
                     title='Number of variables'
                     )

n_select_mp = Slider(start=1,
                     end=2000,
                     step=1,
                     value=500,
                     title='Number of samples'
                     )

# Update the plot when parameters are changed
gen_type_select_mp.on_change('value', update_mp)
N_select_mp.on_change('value', update_mp)
n_select_mp.on_change('value', update_mp)

div_mp = Div(text='<b>Parameters:</b><br>', width=300, height=100)

src_mp, src_pdf_mp = make_dataset_mp(N_select_mp.value, n_select_mp.value, gen_type_select_mp.value, div_mp)

fig_mp = make_plot_mp(src_mp, src_pdf_mp)

controls_mp = WidgetBox(N_select_mp,
                        n_select_mp,
                        gen_type_select_mp,
                        div_mp,
                        width=300,
                        )

# Create a row layout
layout_mp = row(controls_mp, fig_mp)

# Make a tab with the layout

tab_mp = Panel(child=layout_mp, title='Marcenko-Pastur')

### ALL TABS TOGETHER
tabs = Tabs(tabs=[tab_wigner, tab_mp])

curdoc().add_root(tabs)