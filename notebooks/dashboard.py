import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (
    BoxSelectTool,
    ColumnDataSource,
    CustomJS,
    MultiChoice,
    RangeSlider,
    Select,
    CrosshairTool, Slider, LegendItem, Legend, HoverTool, Div, Spacer
)
from bokeh.models import (
    ColorBar,
    GeoJSONDataSource,
    LinearColorMapper,
)
from bokeh.palettes import Light8, Light, Sunset, Sunset11, Light9
from bokeh.plotting import curdoc
from bokeh.plotting import figure


def create_circle_plot(sources, x_default, y_default, palette):
    p = figure(
        title="Circle Plot",
        width=800,
        height=450,
        tools=["pan,wheel_zoom,reset,hover", BoxSelectTool(dimensions="width")],
        active_drag="pan",
    )
    p.xaxis.axis_label = x_default
    p.yaxis.axis_label = y_default

    circle_plots = {}
    legend_items = {}
    for i, country in enumerate(sources):
        circle_plots[country] = p.circle(
            x="x1",
            y="y",
            source=sources[country],
            # legend_label=country,
            fill_color=palette[i],
            size=10,
        )
        legend_items[country] = LegendItem(label=country, renderers=[circle_plots[country]])
    legend = Legend(items=list(legend_items.values()))
    p.add_layout(legend)

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [("Country", "@country"), ("Year", "@year"), ("X", "@x1"), ("Y", "@y")]
    hover.mode = 'mouse'

    return p, circle_plots, legend_items


def create_line_plot(sources, x_default, y_default, palette, y_range):
    p = figure(
        title="Line Plot",
        width=800,
        height=450,
        tools=["pan,wheel_zoom,reset,hover",],
        active_drag="pan",
        y_range=y_range,
    )
    p.xaxis.axis_label = x_default
    p.yaxis.axis_label = y_default

    line_plots = {}
    legend_items = {}
    for i, country in enumerate(sources):
        line_plots[country] = p.line(
            x="x2",
            y="y",
            source=sources[country],
            line_color=palette[i],
            line_width=1,
        )
        legend_items[country] = LegendItem(label=country, renderers=[line_plots[country]])
    legend = Legend(items=list(legend_items.values()))
    p.add_layout(legend)

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [("Country", "@country"), ("Year", "@year"), ("X", "@x2"), ("Y", "@y")]
    hover.mode = 'mouse'

    return p, line_plots, legend_items


def add_vlinked_crosshairs(fig1, fig2):
    # https://stackoverflow.com/questions/37965669/how-do-i-link-the-crosshairtool-in-bokeh-over-several-plots
    js_move = '''if(cb_obj.x >= fig.x_range.start && cb_obj.x <= fig.x_range.end && cb_obj.y >= fig.y_range.start && cb_obj.y <= fig.y_range.end)
                        { cross.spans.height.computed_location = cb_obj.sx }
                     else 
                        { cross.spans.height.computed_location = null }'''
    js_leave = 'cross.spans.height.computed_location = null'

    cross1 = CrosshairTool()
    cross2 = CrosshairTool()
    fig1.add_tools(cross1)
    fig2.add_tools(cross2)
    args = {'cross': cross2, 'fig': fig1}
    fig1.js_on_event('mousemove', CustomJS(args=args, code=js_move))
    fig1.js_on_event('mouseleave', CustomJS(args=args, code=js_leave))
    args = {'cross': cross1, 'fig': fig2}
    fig2.js_on_event('mousemove', CustomJS(args=args, code=js_move))
    fig2.js_on_event('mouseleave', CustomJS(args=args, code=js_leave))


#######################################
# INTERACTIVE SCATTERPLOT
#######################################
def create_interactive_scatterplots(
        df,
        x1_default="co2_emissions_tonnes_per_person",
        x2_default="year",
        y_default="average_annual_temp",
        default_countries=None,
):
    if default_countries is None:
        default_countries = [
            "Austria",
            "Germany",
            "France",
            "Spain",
            "Netherlands",
            "Sweden",
        ]

    initial_data = df[df["country"].isin(default_countries)].sort_values(by="year")
    # create data source
    initial_data_by_country = initial_data.groupby("country").agg(list)

    sources = {}
    for country in initial_data_by_country.index:
        sources[country] = ColumnDataSource({
            "x1": initial_data_by_country.loc[country, x1_default],
            "x2": initial_data_by_country.loc[country, x2_default],
            "y": initial_data_by_country.loc[country, y_default],
            "year": initial_data_by_country.loc[country, "year"],
            "country": np.repeat(country, len(initial_data_by_country.loc[country, x1_default]))
        })

    # CREATE FIGURES
    palette = Light9
    p_circle_plot, circle_plots, circle_plot_legend = create_circle_plot(sources, x1_default, y_default, palette)
    p_line_plot, line_plots, line_plot_legend = create_line_plot(sources, x2_default, y_default, palette,
                                                                 y_range=p_circle_plot.y_range)

    # CREATE WIDGETS
    axis_options = sorted(list(df.columns.values))
    axis_options.remove("country")
    axis_options.remove("iso_a3")
    # Shared Y-axis
    select_y_shared = Select(
        title="Shared y-axis:",
        value=y_default,
        options=axis_options,
        sizing_mode="stretch_width",
        height_policy="min",
    )
    # x-axis scatterplot 1
    select_x1 = Select(
        title="X-axis left:",
        value=x1_default,
        options=axis_options,
        sizing_mode="stretch_width",
        height_policy="min",
    )
    # x-axis scatterplot 2
    select_x2 = Select(
        title="X-axis right:",
        value=x2_default,
        options=axis_options,
        sizing_mode="stretch_width",
        height_policy="min",
    )

    # add country selection
    country_options = sorted(df["country"].unique())
    choice_country = MultiChoice(
        value=default_countries,
        options=country_options,
        max_items=8,
        title="Countries:",
        sizing_mode="stretch_width",
        height_policy="min",
        styles={"color": "black"}
    )
    # add year selection
    date_range_slider = RangeSlider(
        title="Year(s)",
        value=(df["year"].min(), df["year"].max()),
        start=df["year"].min(),
        end=df["year"].max(),
        step=1,
        sizing_mode="stretch_width",
        height_policy="min",
    )

    #######################################

    #######################################
    # CREATE CALLBACKS
    #######################################
    # define function providing filtered data
    def get_filtered_data(
            x1=select_x1.value,
            x2=select_x2.value,
            y=select_y_shared.value,
            countries=choice_country.value,
            date_range=date_range_slider.value,
    ):
        # if indices =
        subset = df[df["country"].isin(set(countries))].sort_values(by="year")
        subset = subset[
            (subset["year"] >= date_range[0]) & (subset["year"] <= date_range[1])
            ]
        subset_by_country = subset.groupby("country").agg(list)
        data_by_country = {}
        for country in subset_by_country.index:
            data_by_country[country] = {
                "x1": subset_by_country.loc[country, x1],
                "x2": subset_by_country.loc[country, x2],
                "y": subset_by_country.loc[country, y],
                "year": subset_by_country.loc[country, "year"],
                "country": np.repeat(country, len(subset_by_country.loc[country, x1]))
            }

        return data_by_country

    def callback_y_shared(attr, old, new):
        new_data = get_filtered_data(
            y=new,
            x1=select_x1.value,
            x2=select_x2.value,
            countries=choice_country.value,
            date_range=date_range_slider.value,
        )
        for country in sources:
            sources[country].data["y"] = new_data[country]["y"]
        # source.data["y"] = new_data["y"]
        p_circle_plot.yaxis.axis_label = new
        p_line_plot.yaxis.axis_label = new
        # p2.xaxis.axis_label = new

    def callback_x1(attr, old, new):
        new_data = get_filtered_data(
            y=select_y_shared.value,
            x1=new,
            x2=select_x2.value,
            countries=choice_country.value,
            date_range=date_range_slider.value,
        )
        for country in sources:
            sources[country].data["x1"] = new_data[country]["x1"]
        p_circle_plot.xaxis.axis_label = new

    def callback_x2(attr, old, new):
        new_data = get_filtered_data(
            y=select_y_shared.value,
            x1=select_x2.value,
            x2=new,
            countries=choice_country.value,
            date_range=date_range_slider.value,
        )
        for country in sources:
            sources[country].data["x2"] = new_data[country]["x2"]
        p_line_plot.xaxis.axis_label = new

    def callback_country(attr, old, new):
        new_data = get_filtered_data(
            y=select_y_shared.value,
            x1=select_x1.value,
            x2=select_x2.value,
            countries=new,
            date_range=date_range_slider.value,
        )

        # delete removed countries
        for country in set(sources.keys()).difference(new):
            print(f"delete {country}")
            p_circle_plot.renderers.remove(circle_plots[country])
            p_circle_plot.legend.items.remove(circle_plot_legend[country])
            p_line_plot.renderers.remove(line_plots[country])
            p_line_plot.legend.items.remove(line_plot_legend[country])

            del sources[country]
            del circle_plots[country]
            del circle_plot_legend[country]
            del line_plots[country]
            del line_plot_legend[country]

        for country in new:
            if country in sources:
                sources[country].data = new_data[country]
            else:
                sources[country] = ColumnDataSource(new_data[country])

                circle_plots[country] = p_circle_plot.circle(
                    x="x1",
                    y="y",
                    source=sources[country],
                    # legend_label=country,
                    fill_color=palette[len(sources)],
                    size=10,
                )
                circle_plot_legend[country] = LegendItem(label=country, renderers=[circle_plots[country]])
                p_circle_plot.legend.items = list(circle_plot_legend.values())

                line_plots[country] = p_line_plot.line(
                    x="x2",
                    y="y",
                    source=sources[country],
                    line_color=palette[len(sources)],
                    line_width=1,
                )
                line_plot_legend[country] = LegendItem(label=country, renderers=[line_plots[country]])
                p_line_plot.legend.items = list(line_plot_legend.values())

    def callback_date(attr, old, new):
        new_data = get_filtered_data(
            y=select_y_shared.value,
            x1=select_x1.value,
            x2=select_x2.value,
            countries=choice_country.value,
            date_range=new,
        )
        for country in sources:
            sources[country].data = new_data[country]

    # add callbacks to widgets
    select_y_shared.on_change("value", callback_y_shared)
    select_x1.on_change("value", callback_x1)
    select_x2.on_change("value", callback_x2)
    choice_country.on_change("value", callback_country)
    date_range_slider.on_change("value", callback_date)

    def callback(attr, old, new, country):
        for country in sources:
            source = sources[country]
            try:
                selections = new.indices
                select_inds = [selections[0]]
                if len(selections) == 1:
                    selected_issuer = source.data['y'][selections[0]]
                    for i in range(0, len(source.data['x'])):
                        if i != selections[0]:
                            issuer = source.data['y'][i]
                            if issuer == selected_issuer:
                                select_inds.append(i)
                if len(selections) == 0:
                    for i in range(0, len(source.data['x'])):
                        select_inds.append(i)
                new.indices = select_inds
            except IndexError:
                pass

    shared_widgets = column(
        date_range_slider, select_y_shared, select_x1, select_x2, choice_country
    )
    return shared_widgets, p_circle_plot, p_line_plot


def create_map_histogram(df, default_var="average_annual_temp", default_year=2015):
    def get_filtered_data(variable=default_var, year=default_year):
        # filter according to year
        subset = df[df["year"] == year]

        # make data for map
        subset_map = subset[[variable, "country", "iso_a3"]]
        subset_map = subset_map.dropna()
        map_low_high = (subset_map[variable].min(), subset_map[variable].max())
        subset_map = subset_map.rename(columns={variable: "map_variable"})
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        merged = world.merge(subset_map, on="iso_a3")
        merged_json = json.loads(merged.to_json())
        json_map = json.dumps(merged_json)

        # make data for histogram
        hist, hist_edges = np.histogram(subset[variable], bins="auto")
        # create data source
        dict_histogram = {"hist": hist, "left": hist_edges[:-1], "right": hist_edges[1:]}

        return json_map, map_low_high, dict_histogram

    initial_json_map, initial_map_low_high, initial_dict_histogram = get_filtered_data()

    # create data source
    source_map = GeoJSONDataSource(geojson=initial_json_map)
    source_histogram = ColumnDataSource(
        initial_dict_histogram
    )

    # Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
    color_mapper = LinearColorMapper(
        palette="Magma256",
        low=initial_map_low_high[0],
        high=initial_map_low_high[1],
    )

    p_map = create_choropleth_map(source_map, color_mapper)
    p_hist = create_histogram(source_histogram)

    p_hist.xaxis.axis_label = default_var

    axis_options = sorted(list(df.columns.values))
    axis_options.remove("cc_somw_t_per")
    axis_options.remove("cc_vser_somw_t_per")
    axis_options.remove("cc_vser_t_per")
    axis_options.remove("country")
    axis_options.remove("gdp_per_capita_yearly_growth")
    axis_options.remove("inflation_annual_percent")
    axis_options.remove("iso_a3")
    axis_options.remove("year")

    # add variable selection
    select_variable = Select(
        title="Variable:",
        value=default_var,
        options=axis_options,
        sizing_mode="stretch_width",
        height_policy="min",
    )

    # add year selection
    date_slider = Slider(
        title="Year",
        value=default_year,
        start=df["year"].min(),
        end=df["year"].max(),
        step=1,
        sizing_mode="stretch_width",
        height_policy="min",
    )

    # ADD CALLBACKS
    def callback_variable(attr, old, new):
        new_data_map, new_map_low_high, new_data_histogram = get_filtered_data(new, date_slider.value)
        # update map
        source_map.geojson = new_data_map
        color_mapper.low = new_map_low_high[0]
        color_mapper.high = new_map_low_high[1]
        # update hist
        source_histogram.data = new_data_histogram
        p_hist.xaxis.axis_label = new

    def callback_date(attr, old, new):
        new_data_map, _, new_data_histogram = get_filtered_data(select_variable.value, new)
        source_map.geojson = new_data_map
        source_histogram.data = new_data_histogram

    # add callbacks to widgets
    select_variable.on_change("value", callback_variable)
    date_slider.on_change("value", callback_date)

    shared_widgets = column(date_slider, select_variable)
    return shared_widgets, p_map, p_hist


def create_histogram(source):
    initial_data = df

    # CREATE FIGURE
    p = figure(
        title="Histogram",
        width=800,
        height=450,
        tools=["pan,wheel_zoom,box_select,reset,hover", BoxSelectTool(dimensions="width")],
        active_drag="pan",
    )

    # PLOT HISTOGRAM
    p.quad(
        bottom=0,
        top="hist",
        left="left",
        right="right",
        fill_color="#2596be",
        line_color="black",
        source=source,
    )

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [("Value", "@hist"), ("Left", "@left"), ("Right", "@right")]
    hover.mode = 'mouse'

    return p


def create_choropleth_map(source, color_mapper):
    # Create color bar.
    color_bar = ColorBar(
        color_mapper=color_mapper,
        label_standoff=8,
        width=500,
        height=20,
        border_line_color=None,
        location=(0, 0),
        orientation="horizontal",
    )

    # Create figure object.
    p = figure(
        title="Choropleth Map",
        height=450,
        width=800,
        tools=["pan,wheel_zoom,reset,hover"]
    )

    p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
    p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels

    # Add patch renderer to figure.
    p.patches(
        "xs",
        "ys",
        source=source,
        fill_color={"field": "map_variable", "transform": color_mapper},
        line_color="black",
        line_width=0.25,
        fill_alpha=1,

    )
    # Specify figure layout.
    p.add_layout(color_bar, "below")

    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [("Country", "@country"), ("Value", "@map_variable")]
    hover.mode = 'mouse'

    return p


# MAIN
# CONSTANTS
PROCESSED_DATA_PATH = "../data/processed"
RANDOM_STATE = 42

# LOAD DATA
df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "wrangled_data.csv"))
df = df.round(2)

# CREATE MAP AND HISTOGRAM
widgets_map_histogram, cho_map, histogram = create_map_histogram(df)

# CREATE CIRCLE AND LINE PLOT
widgets_scatter_line, circle_plot, line_plot = create_interactive_scatterplots(df)

# SETUP LAYOUT
l = row(
    column(Div(text="""<h2>Inputs</h2><h3>Map/Histogram:</h3>""", ),
           widgets_map_histogram,
           Div(text="""<br><br><hr style="width: 250px;"><br><br>""", ),
           Div(text="""<h3>Circle/Line:</h3>""", ),
           widgets_scatter_line, sizing_mode="fixed", width=250,
           styles={"height": "100%", "background-color": "#181c1c", "padding": "20px", "color": "white"}),
    column(cho_map, circle_plot, sizing_mode="stretch_both"),
    column(histogram, line_plot, sizing_mode="stretch_both"),
    sizing_mode="stretch_both", styles={"background-color": "#181c1c"})

curdoc().theme = 'dark_minimal'
curdoc().add_root(
    Div(text="""<h1>Climate Dashboard</h1>""",
        styles={"width": "100%", "background-color": "#2596be", "padding": "10px", "margin": "0", "color": "white"}))
curdoc().add_root(l, )
