import numpy as np
from scipy.stats import cauchy, norm, semicircular, uniform
import pandas as pd

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Panel
from bokeh.models.widgets import Div, Slider, Tabs
from bokeh.layouts import row, WidgetBox
from bokeh.plotting import figure


MAX_GEN_TYPE = 3


def gen_type_mgr(gen_type: int, check_moments=False):
    if gen_type == 1:
        gen = uniform(
            loc=-uniform.mean(scale=1/uniform.std()),
            scale=1/uniform.std()
        )
        gen_type_str = 'Uniform'
    elif gen_type == 2:
        gen = norm
        gen_type_str = 'Gaussian'
    elif gen_type == 3:
        gen = cauchy
        gen_type_str = 'Cauchy'
    else:
        raise ValueError('Unknown generator type')

    if check_moments:
        assert gen.mean() == 0.0, 'Random generator should be centered.'
        assert gen.std() == 1.0, 'Random generator should be of unit variance.'

    return gen, gen_type_str


def make_dataset_wigner(N, gen_type, div_):
    """Creates a ColumnDataSource object with data to plot.
    """
    gen, gen_type_str = gen_type_mgr(gen_type)

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

    params_text = '<b>Parameters:</b><br><ul><li>Entries distribution \
    = {:s}</li></ul>'.format(gen_type_str)
    div_.text = params_text

    # Convert dataframe to column data source
    return ColumnDataSource(df), ColumnDataSource(df_pdf)


def make_dataset_mp(N, n, gen_type, div_):
    """Creates a ColumnDataSource object with data to plot.
    """
    gen, gen_type_str = gen_type_mgr(gen_type)

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

    def marcenko_pastur_pdf(x: float):
        return np.sqrt(
            (lambda_plus - x) * (x - lambda_minus)
            ) / (2*np.pi*c*x) if x > lambda_minus and x < lambda_plus else 0.0

    pdf = list(map(marcenko_pastur_pdf, xx))
    df_pdf = pd.DataFrame({'xx': xx, 'pdf': pdf})

    params_text = '<b>Parameters:</b><br><ul><li>Entries distribution \
    = {:s}</li></ul>'.format(gen_type_str)
    div_.text = params_text

    # Convert dataframe to column data source
    return ColumnDataSource(df), ColumnDataSource(df_pdf)


def make_dataset_large_cov(N,
                           n,
                           gen_type,
                           eig1,
                           freq1,
                           eig2,
                           freq2,
                           eig3,
                           freq3,
                           div_
                           ):
    """Creates a ColumnDataSource object with data to plot.
    """
    gen, gen_type_str = gen_type_mgr(gen_type)

    std_pop = np.array([eig1, eig2, eig3])
    total = freq1+freq2+freq3
    freq1 = freq1/total
    freq2 = freq2/total
    freq3 = freq3/total
    std_pop_distr = np.array([freq1, freq2, freq3])
    cov_pop_sqrt = np.diag(
        np.random.choice(std_pop, p=std_pop_distr, size=N) ** (0.5)
        )

    X = gen.rvs(size=N*n).reshape(N, n)
    Y = np.dot(cov_pop_sqrt, X)
    A = np.dot(Y, Y.T)/n

    spectrum = np.linalg.eigvalsh(A)
    hist, edges = np.histogram(spectrum, density=True, bins=50)

    df = pd.DataFrame({'top': hist,
                       'bottom': 0,
                       'left': edges[:-1],
                       'right': edges[1:],
                       })

    df_eig = pd.DataFrame({'top': std_pop_distr*3,
                           'bottom': 0,
                           'left': std_pop-0.002,
                           'right': std_pop+0.002,
                           })

    params_text = '<b>Parameters:</b><br><ul><li>Entries distribution \
    = {:s}</li></ul>'.format(gen_type_str)
    div_.text = params_text

    # Convert dataframe to column data source
    return ColumnDataSource(df), ColumnDataSource(df_eig)


def make_dataset_general(N, gen_type, div_):
    """Creates a ColumnDataSource object with data to plot.
    """
    gen, gen_type_str = gen_type_mgr(gen_type)

    A = gen.rvs(size=N**2).reshape(N, N) / np.sqrt(N)

    spectrum = np.linalg.eigvals(A)
    x, y = np.real(spectrum), np.imag(spectrum)

    df = pd.DataFrame({'x': x,
                       'y': y,
                       })

    params_text = '<b>Parameters:</b><br><ul><li>Entries distribution \
    = {:s}</li></ul>'.format(gen_type_str)
    div_.text = params_text

    # Convert dataframe to column data source
    return ColumnDataSource(df)


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


def make_plot_large_cov(src_large_cov, src_eig_large_cov):
    """Create a figure object to host the plot.
    """
    # Blank plot with correct labels
    fig_large_cov = figure(plot_width=700,
                           plot_height=700,
                           title='Large covariance matrix',
                           )

    fig_large_cov.quad(top='top',
                       bottom='bottom',
                       left='left',
                       right='right',
                       source=src_large_cov,
                       fill_color='navy',
                       line_color='white',
                       alpha=0.5,
                       )

    fig_large_cov.quad(top='top',
                       bottom='bottom',
                       left='left',
                       right='right',
                       source=src_eig_large_cov,
                       fill_color='red',
                       line_color='black',
                       legend='Covariance eigenvalues',
                       alpha=1,
                       )

    # Continuous pdf
    # fig_large_cov.legend.click_policy = 'hide'
    # fig_large_cov.legend.location = 'top_right'
    return fig_large_cov


def make_plot_general(src_general):
    """Create a figure object to host the plot.
    """
    # Blank plot with correct labels
    fig_general = figure(plot_width=700,
                         plot_height=700,
                         title="General case of matrix rescaled by 1/sqrt(N)",
                         )

    theta = np.linspace(0, 2*np.pi, 100)
    x_circle, y_circle = np.cos(theta), np.sin(theta)

    fig_general.scatter('x', 'y', source=src_general, color='navy', alpha=0.7)
    fig_general.line(x_circle, y_circle, color='red', alpha=0.9)

    # fig_general.legend.click_policy = 'hide'
    # fig_general.legend.location = 'top_right'
    return fig_general


def update_wigner(attr, old, new):
    """Update ColumnDataSource object.
    """
    # Change to selected values
    gen_type = gen_type_select_wigner.value
    N = N_select_wigner.value

    # Create new ColumnDataSource
    new_src_wigner, new_src_pdf_wigner = make_dataset_wigner(N,
                                                             gen_type,
                                                             div_wigner
                                                             )

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
    new_src_mp, new_src_pdf_mp = make_dataset_mp(N, n, gen_type, div_mp)

    # Update the data on the plot
    src_mp.data.update(new_src_mp.data)
    src_pdf_mp.data.update(new_src_pdf_mp.data)


def update_large_cov(attr, old, new):
    """Update ColumnDataSource object.
    """
    # Change to selected values
    gen_type = gen_type_select_large_cov.value
    N = N_select_large_cov.value
    n = n_select_large_cov.value
    eig1 = eig_1_select_large_cov.value
    freq1 = eig_1_freq_select_large_cov.value
    eig2 = eig_2_select_large_cov.value
    freq2 = eig_2_freq_select_large_cov.value
    eig3 = eig_3_select_large_cov.value
    freq3 = eig_3_freq_select_large_cov.value

    # Create new ColumnDataSource
    new_src_large_cov, new_src_eig_large_cov \
        = make_dataset_large_cov(N,
                                 n,
                                 gen_type,
                                 eig1,
                                 freq1,
                                 eig2,
                                 freq2,
                                 eig3,
                                 freq3,
                                 div_large_cov,
                                 )

    # Update the data on the plot
    src_large_cov.data.update(new_src_large_cov.data)
    src_eig_large_cov.data.update(new_src_eig_large_cov.data)


def update_general(attr, old, new):
    """Update ColumnDataSource object.
    """
    # Change to selected values
    gen_type = gen_type_select_general.value
    N = N_select_general.value

    # Create new ColumnDataSource
    new_src_general = make_dataset_general(N, gen_type, div_general)

    # Update the data on the plot
    src_general.data.update(new_src_general.data)


######################################################################
# WIGNER
######################################################################
# Slider to select parameters
gen_type_select_wigner = Slider(start=1,
                                end=MAX_GEN_TYPE,
                                step=1,
                                title='Entries distribution',
                                value=1,
                                )

N_select_wigner = Slider(start=1,
                         end=2000,
                         step=1,
                         value=500,
                         title='Number of variables'
                         )

# Update the plot when parameters are changed
gen_type_select_wigner.on_change('value', update_wigner)
N_select_wigner.on_change('value', update_wigner)

div_wigner = Div(text='<b>Parameters:</b><br>', width=300, height=100)

src_wigner, src_pdf_wigner = make_dataset_wigner(N_select_wigner.value,
                                                 gen_type_select_wigner.value,
                                                 div_wigner
                                                 )

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
# MARCENKO-PASTUR
######################################################################
# Slider to select parameters
gen_type_select_mp = Slider(start=1,
                            end=MAX_GEN_TYPE,
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

src_mp, src_pdf_mp = make_dataset_mp(N_select_mp.value,
                                     n_select_mp.value,
                                     gen_type_select_mp.value,
                                     div_mp
                                     )

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


######################################################################
# LARGE COVARIANCE MATRIX
######################################################################
# Slider to select parameters
gen_type_select_large_cov = Slider(start=1,
                                   end=MAX_GEN_TYPE,
                                   step=1,
                                   title='Entries distribution',
                                   value=1,
                                   )

N_select_large_cov = Slider(start=1,
                            end=2000,
                            step=1,
                            value=100,
                            title='Number of variables'
                            )

n_select_large_cov = Slider(start=1,
                            end=2000,
                            step=1,
                            value=1000,
                            title='Number of samples'
                            )

eig_1_select_large_cov = Slider(start=0,
                                end=1,
                                step=0.05,
                                value=0.2,
                                title='Covariance eigenvalue 1'
                                )

eig_1_freq_select_large_cov = Slider(start=0,
                                     end=1,
                                     step=0.05,
                                     value=0.33,
                                     title='Frequency'
                                     )

eig_2_select_large_cov = Slider(start=0,
                                end=1,
                                step=0.05,
                                value=0.5,
                                title='Covariance eigenvalue 2'
                                )

eig_2_freq_select_large_cov = Slider(start=0,
                                     end=1,
                                     step=0.05,
                                     value=0.33,
                                     title='Frequency'
                                     )

eig_3_select_large_cov = Slider(start=0,
                                end=1,
                                step=0.05,
                                value=0.8,
                                title='Covariance eigenvalue 3'
                                )

eig_3_freq_select_large_cov = Slider(start=0,
                                     end=1,
                                     step=0.05,
                                     value=0.33,
                                     title='Frequency'
                                     )

# Update the plot when parameters are changed
gen_type_select_large_cov.on_change('value', update_large_cov)
N_select_large_cov.on_change('value', update_large_cov)
n_select_large_cov.on_change('value', update_large_cov)

eig_1_select_large_cov.on_change('value', update_large_cov)
eig_1_freq_select_large_cov.on_change('value', update_large_cov)

eig_2_select_large_cov.on_change('value', update_large_cov)
eig_2_freq_select_large_cov.on_change('value', update_large_cov)

eig_3_select_large_cov.on_change('value', update_large_cov)
eig_3_freq_select_large_cov.on_change('value', update_large_cov)

div_large_cov = Div(text='<b>Parameters:</b><br>', width=300, height=100)

src_large_cov, src_eig_large_cov \
    = make_dataset_large_cov(N_select_large_cov.value,
                             n_select_large_cov.value,
                             gen_type_select_large_cov.value,
                             eig_1_select_large_cov.value,
                             eig_1_freq_select_large_cov.value,
                             eig_2_select_large_cov.value,
                             eig_2_freq_select_large_cov.value,
                             eig_3_select_large_cov.value,
                             eig_3_freq_select_large_cov.value,
                             div_large_cov,
                             )

fig_large_cov = make_plot_large_cov(src_large_cov, src_eig_large_cov)

controls_large_cov = WidgetBox(N_select_large_cov,
                               n_select_large_cov,
                               gen_type_select_large_cov,
                               eig_1_select_large_cov,
                               eig_1_freq_select_large_cov,
                               eig_2_select_large_cov,
                               eig_2_freq_select_large_cov,
                               eig_3_select_large_cov,
                               eig_3_freq_select_large_cov,
                               div_large_cov,
                               width=300,
                               )

# Create a row layout
layout_large_cov = row(controls_large_cov, fig_large_cov)

# Make a tab with the layout

tab_large_cov = Panel(child=layout_large_cov, title='Large covariance matrix')


######################################################################
# GENERAL CASE (Non-Hermitian matrix rescaled by 1/sqrt(N))
######################################################################
# Slider to select parameters
gen_type_select_general = Slider(start=1,
                                 end=MAX_GEN_TYPE,
                                 step=1,
                                 title='Entries distribution',
                                 value=1,
                                 )

N_select_general = Slider(start=1,
                          end=2000,
                          step=1,
                          value=500,
                          title='Number of variables'
                          )

# Update the plot when parameters are changed
gen_type_select_general.on_change('value', update_general)
N_select_general.on_change('value', update_general)

div_general = Div(text='<b>Parameters:</b><br>', width=300, height=100)

src_general = make_dataset_general(N_select_general.value,
                                   gen_type_select_general.value,
                                   div_general
                                   )

fig_general = make_plot_general(src_general)

controls_general = WidgetBox(N_select_general,
                             gen_type_select_general,
                             div_general,
                             width=300,
                             )

# Create a row layout
layout_general = row(controls_general, fig_general)

# Make a tab with the layout

tab_general = Panel(child=layout_general, title='General case')


# ALL TABS TOGETHER
tabs = Tabs(tabs=[tab_wigner, tab_mp, tab_large_cov, tab_general])

curdoc().add_root(tabs)
