"""
Functions for plotting graphs.
"""

from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.subplots as sp
from IPython.display import display, HTML
import re
from . import utils

# other settings
pio.templates.default = "plotly_white"

legend_style = dict(
            orientation="h",  # Horizontal orientation
            x=0.0,            # Center horizontally
            y=-0.2,           # Position below the plot area
            xanchor="left",
            yanchor="top",
        )

COLOR_PRIMARY = "#1CABE2"

PALLETE_QUALITATIVE = [
    #"#0058AB",  # Deep Blue
    '#374EA2',   # Medium Blue/Purple-ish
    "#1CABE2",  # Bright Cyan/Light Blue
    "#00833D",  # Dark Green
    "#80BD41",  # Lime Green
    "#6A1E74",  # Dark Purple
    "#961A49",  # Deep Maroon
    "#E2231A",  # Bright Red
    "#F26A21",  # Bright Orange
    "#FFC20E",  # Bright Yellow
    "#FFF09C",  # Pale Yellow/Cream
    "#9E9E9E",  # Medium Gray
    "#231f20",  # Raisin Black

]


PALLETE_GRAYS = ["#333333", "#6B6B6B", "#B3B3B3", "#D7D7D7", "#E8E8E8"]
MODEBAR_CONFIG = {
    "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "autoScale2d"],
    "displaylogo": False,
}
MARGINS = {"l": 50, "r": 50, "b": 100, "t": 100, "pad": 4}


# Mapping rating values to numerical
rating_mapping = {
    'Fully Achieved': 1,
    'Partially Achieved': 0.75,
    'Not Achieved': 0.5,
    'No Data': 0
}

# dictionary matching Unicef's SDG color-scheme to SDG goal number
SDG_color_map = {
    '1':      '#E5243B',  # No Poverty
    '2':      '#DDA63A',  # Zero Hunger
    '3':      '#4C9F38',  # Good Health and Well-being
    '4':      '#C5192D',  # Quality Education
    '5':      '#FF3A21',  # Gender Equality
    '6':      '#26BDE2',  # Clean Water and Sanitation
    '7':      '#FCC30B',  # AFFORDABLE AND CLEAN ENERGY
    '8':      '#A21942',  # DECENT WORK AND ECONOMIC GROWTH
    '9':      '#FD6925',  # Industry, Innovation & Infrastructure
    '10':     '#DD1367',  # Reduce inequality within and among countries 
    '11':     '#FD9D24',  # SUSTAINABLE CITIES AND COMMUNITIES
    '12':     '#BF8B2E',  # RESPONSIBLE CONSUMPTION & PRODUCTION
    '13':     '#3F7E44',  # Climate Change
    '14':     '#0A97D9',  # LIFE BELOW WATER
    '15':     '#56C02B',  # LIFE ON LAND
    '16':     '#00689D',  # Peace, Justice and Strong Institutions
    '17':     '#19486A',  # Partnerships for the Goals
    'Unknown': "#9E9E9E",  # Unknown (Medium Gray, not SDG colour)
    'Undefined': "#9E9E9E", # Undefined (Medium Gray, not SDG colour)
    'Multiple': '#333333',  # Multiple
}

# dictionary matching Unicef's SDG goal number to description of goal
SDG_goals = {
    '1':      'No Poverty', 
    '2':      'Zero Hunger',  
    '3':      'Good Health and Well-being',  
    '4':      'Quality Education', 
    '5':      'Gender Equality', 
    '6':      'Clean Water and Sanitation', 
    '7':      'Affordable and Clean Energy', 
    '8':      'Decent Jobs and Economic Growth',  
    '9':      'Industry, Innovation and Infrastructure', 
    '10':     'Reduced Inequalities ',  
    '11':     'Sustainable Cities and Communities', 
    '12':     'Responsible Consumption and Production', 
    '13':     'Climate Action',  
    '14':     'Life Below Water', 
    '15':     'Life on Land',  
    '16':     'Peace and Justice - Strong Institutions',  
    '17':     'Partnerships for the Goals', 
    'Unknown': 'Unknown', 
    'Undefined': 'Undefined',    
    'Multiple': 'Multiple', 
}

goal_area_colors = {

    'Survive and Thrive': "#0058AB",  # Deep Blue
    'Protection from Violence and Exploitation': "#1CABE2",  # Bright Cyan/Light Blue
    'Safe and Clean Environment': "#00833D",  # Dark Green
    'Equitable Chance in Life': "#80BD41",  # Lime Green
    'Cross Sectoral': "#6A1E74",  # Dark Purple
    'Learn': "#961A49",  # Deep Maroon
    'Management': "#E2231A",  # Bright Red
    'UN Coordination': "#F26A21",  # Bright Orange
    'Special Purpose': "#FFC20E",  # Bright Yellow
    'Development Effectiveness': "#FFF09C",  # Pale Yellow/Cream
    'Independent oversight and assurance': "#9E9E9E",  # Medium Gray

    'Other': PALLETE_GRAYS[0],

    'Systems strengthening to leave no one behind': "#00BCD4",  # Cyan
    'Advocacy and communications': "#6A2976",  # Purple,
    'Data, research, evaluation and knowledge management': "#FFC107",  # Amber,
    'Partnerships and engagement: public and private': "#FF5722",  # Deep Orange,
    'Community engagement, social and behaviour change': "#3F51B5",  # Indigo, 
    'Innovation': "#008545",  # Dark Green,
    'Unknown': "#9E9E9E",  # Grey,
    'Risk-informed humanitarian and development nexus programming': "#E91E63",  # Pink,
    'Digital transformation': "#971B4C",  # Dark Red

}

partner_types = ['Civil Society Organization',  'Financial Service Provider',  'Government',  'Multi-Lateral Organization',  'No Partner Type Indicated',  'Private Sector',  'Un Agency']
partner_types_color = {
    t: c for t, c in zip(partner_types, PALLETE_QUALITATIVE)
}

partners_sub_types = ['Civil Society Organization',  'Civil Society Organization - Academic Institution',  'Civil Society Organization - Community Based Organization',  'Civil Society Organization - International Ngo',  'Civil Society Organization - National Ngo',  'Civil Society Organization - Red Cross/Red Crescent National Societies',  'Financial Service Provider',  'Government',  'Multi-Lateral Organization',  'No Partner Type Indicated',  'Private Sector',  'Un Agency']
partner_sub_types_colors = {
    t: c for t, c in zip(partners_sub_types, PALLETE_QUALITATIVE)
}


def hex_to_rgba(hex_color, alpha=0.5):
    hex_color = hex_color.lstrip('#')  # Remove the '#' character
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))  # Convert hex to RGB
    return f'rgba({r},{g},{b},{alpha})' 

def plot_funds_total(df_funds: pd.DataFrame, normalise: bool = False):
    if normalise:
        df_funds["value"] = df_funds.groupby("year")["value"].transform(
            lambda x: x / x.sum()
        )
    fig = px.area(
        data_frame=df_funds.groupby(["year", "source"], as_index=False).sum(),
        x="year",
        y="value",
        color="source",
        color_discrete_sequence=PALLETE_QUALITATIVE,
        markers=True,
        labels={
            "year": "Reporting Year",
            "value": (
                "Funds Utilization in USD"
                if not normalise
                else "Funds Utilization as a Share of Yearly Spending"
            ),
            "source": "Funding Source",
        },
    )
    fig.update_yaxes(matches=None, tickformat=".1%" if normalise else None)
    fig.update_layout(
        legend=legend_style,
        legend_title_text="",
    )
    return fig


def plot_funds_disaggregated(
    df_funds: pd.DataFrame,
    normalise: bool = False,
    level: Literal["domain", "area"] = "area",
    legend_column_width: int = 70,
    legend_columns: int = 2,
):
    if normalise:
        df_funds["value"] = df_funds.groupby(["source", "year"])["value"].transform(
            lambda x: x / x.sum()
        )
    if level == "domain":
        mapping = dict(
            zip(
                utils.get_pidb_entries(level="domain", kind="code"),
                utils.get_pidb_entries(level="domain", kind="both"),
            )
        )
        df_funds["results_area"] = df_funds["results_area"].apply(
            lambda x: mapping.get(x.split("-")[0], "Other")
        )
        df_funds = df_funds.groupby(
            ["results_area", "year", "source"], as_index=False
        ).agg({"value": "sum"})
    elif level == "area":
        pass
    else:
        raise ValueError(f"Unknown `level` value {level}.")
    df_funds["results_area"] = df_funds["results_area"].apply(
        lambda x: utils.wrap_text_with_br(x, width=legend_column_width)
    )
    fig = px.area(
        data_frame=df_funds.sort_values(["results_area", "year"]),
        x="year",
        y="value",
        color="results_area",
        color_discrete_sequence=PALLETE_QUALITATIVE,
        markers=True,
        facet_row="source",
        labels={
            "year": "Reporting Year",
            "value": (
                "Utilization in USD"
                if not normalise
                else "Funds Utilization as a Share of Yearly Spending"
            ),
            "results_area": "Results Area",
        },
        height=1300,
    )
    fig.update_yaxes(matches=None)
    fig.update_xaxes(type="category")
    fig.for_each_yaxis(lambda a: a.update(tickformat=".1%" if normalise else None))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig.update_layout(
        legend=legend_style,
        legend_title_text="",
    )
    return fig


def plot_heatmap(df: pd.DataFrame, width: int = None, height: int = None, **kwargs):
    fig = px.imshow(img=df, text_auto=".1%", width=width, height=height, **kwargs)
    return fig


def plot_partnerships(df: pd.DataFrame):
    fig = px.bar(
        data_frame=df,
        x="year",
        y="number_of_agreements",
        text_auto=True,
        labels={
            "year": "Year (Partnership Started)",
            "number_of_agreements": "Number of Agreements",
        },
    )
    fig.update_traces(marker_color=COLOR_PRIMARY)
    fig.update_layout(
        barmode="group",
        xaxis={"type": "category"},
        legend=legend_style,
        legend_title_text="",
    )
    return fig


def plot_hr_by_gender(df: pd.DataFrame, by: str):
    fig = px.bar(
        data_frame=df.groupby([by, "gender"], as_index=False).size(),
        x=by,
        y="size",
        color="gender",
        color_discrete_sequence=PALLETE_QUALITATIVE,
        labels={
            by: by.replace("_", " ").title(),
            "size": "Number of People",
            "gender": "Gender",
        },
    )
    fig.update_layout(
        barmode="group",
        legend=legend_style,
        legend_title_text="",
    )
    return fig


def plot_hr_by_area(df: pd.DataFrame):
    fig = px.bar(
        data_frame=df.groupby(["functional_area", "category"], as_index=False).size(),
        x="functional_area",
        y="size",
        color="category",
        color_discrete_sequence=PALLETE_QUALITATIVE,
        labels={
            "functional_area": "Functional Area",
            "size": "Number of People",
            "category": "Staff Category",
        },
    )
    fig.update_layout(
        legend=legend_style,
        legend_title_text="",
        barmode="group",
    )
    return fig


def plot_top_partners(df: pd.DataFrame, n: int = 10):
    df = df.copy()
    df["vendor"] = df["vendor"].apply(lambda x: utils.wrap_text_with_br(x, width=50))
    fig = px.bar(
        data_frame=df.sort_values("transfers_to_ip").tail(n),
        x="transfers_to_ip",
        y="vendor",
        labels={
            "transfers_to_ip": "Total Transfers to IP (2014-2023)",
            "vendor": "Vendor",
        },
    )
    fig.update_traces(marker_color=COLOR_PRIMARY)
    return fig


def plot_partnerships_by_area(df: pd.DataFrame, legend_columns: int = 2):
    fig = px.area(
        data_frame=df,
        x="fr_start_year",
        y="transfers_to_ip",
        color="goal_area",
        color_discrete_sequence=PALLETE_QUALITATIVE,
        markers=True,
        labels={
            "fr_start_year": "Year",
            "transfers_to_ip": "Transfers to IP",
            "goal_area": "Results Area",
        },
    )
    fig.update_layout(
        xaxis={"type": "category"},
        legend=legend_style,
        legend_title_text="",
        barmode="group",
    )
    return fig


def plot_ram_indicators(df: pd.DataFrame):
    indicator = df["indicator"].unique().item()
    unit = df["indicator_unit"].unique().item()
    if unit != "PERCENT":
        suffix = ""
        ymax = None
    else:
        suffix = "[ in %]"
        ymax = 100
    fig = px.bar(
        data_frame=df.sort_values("target_year"),
        x="target_year",
        y="value",
        color="variable",
        color_discrete_sequence=PALLETE_QUALITATIVE,
        labels={
            "target_year": "Year",
            "value": f"Indicator Value {suffix}",
            "variable": "Value Type",
        },
        title=utils.wrap_text_with_br(indicator, width=80),
    )
    fig.update_layout(
        barmode="group",
        yaxis={"range": [0, ymax]},
        margin=MARGINS,
    )
    return fig


def plot_trips(df: pd.DataFrame):
    fig = px.area(
        data_frame=df,
        x="date",
        y="amount",
        color="reason",
        color_discrete_sequence=PALLETE_QUALITATIVE,
        markers=True,
        labels={
            "date": "Trip Start Date",
            "amount": "Expenses in USD",
            "reason": "Trip Reason",
        },
        title="Monthly Expenses on Travel",
    )
    fig.update_layout(
        legend=legend_style,
        legend_title_text="",
    )
    return fig


def plot_funds_by_cost_category(df: pd.DataFrame, legend_columns: int = 2):
    fig = px.area(
        data_frame=df,
        x="year",
        y="value",
        color="cost_category",
        color_discrete_sequence=PALLETE_QUALITATIVE,
        markers=True,
        labels={
            "year": "Year",
            "value": "Funds Utilization in USD",
            "cost_category": "Cost Category",
        },
    )
    fig.update_layout(
        legend=legend_style,
        legend_title_text="",
    )
    return fig


def plot_treemap(df: pd.DataFrame, path: list[str], **kwargs):
    df = df.query("value > 0").copy()
    for column in path:
        df[column] = df[column].apply(utils.wrap_text_with_br)

    fig = px.treemap(
        data_frame=df,
        path=path,
        values="value",
        color_discrete_sequence=PALLETE_QUALITATIVE,
        **kwargs,
    )
    fig.update_traces(root_color=PALLETE_GRAYS[-1])
    fig.update_traces(
        texttemplate="%{label}<br>%{value:$,.0f} (%{percentParent})",
        selector={"type": "treemap"},
    )
    fig.update_layout(
        margin={
            "t": 50,
            "l": 25,
            "r": 25,
            "b": 25,
        },
    )
    return fig


def plot_outputs(df: pd.DataFrame, title: str = ""):
    fig = px.bar(
        data_frame=df.sort_values(["year", "output"]),
        x="year",
        y="value",
        color="output",
        color_discrete_sequence=PALLETE_QUALITATIVE,
        barmode="stack",
        labels={
            "year": "Year",
            "value": "Funds Utilization in USD",
            "output": "Output",
        },
        title=title,
        height=400,
    )
    fig.update_layout(xaxis={"type": "category"})
    return fig


def plot_transmonee(df: pd.DataFrame):
    disaggregations = ["sex", "residence"]
    measure = df["measure"].unique().item()
    title = df["indicator_name"].unique().item()
    df_temp = df[disaggregations].nunique()
    disaggregations = df_temp[df_temp > 1].index.tolist()
    if not disaggregations:
        color, facet_row = None, None
        plot_fn = px.area
        kwargs = {"markers": True}
    elif len(disaggregations) == 1:
        color = disaggregations[0]
        facet_row = None
        plot_fn = px.bar
        kwargs = {"barmode": "group"}
    else:
        color = "sex"
        facet_row = "residence"
        plot_fn = px.bar
        kwargs = {"barmode": "group"}
    fig = plot_fn(
        data_frame=df,
        x="year",
        y="value",
        color=color,
        color_discrete_sequence=PALLETE_QUALITATIVE,
        facet_row=facet_row,
        **kwargs,
        labels={
            "year": "Year",
            "value": f"Value [{measure}]",
            "sex": "Sex",
        },
        height=None if facet_row is None else 600,
        title=utils.wrap_text_with_br(title, width=70),
    )
    fig.update_xaxes(categoryorder="category ascending")
    fig.update_yaxes(matches=None)
    fig.update_layout(margin=MARGINS)
    return fig



def generate_timeline_from_df(df):
    output = "<div class='container'>\n"
    
    current_year = None
    current_goal_area = None
    
    for _, row in df.iterrows():
        year = row['year']
        goal_area = row['goal_area']
        description = row['need']
        
        # Check if we are in a new year
        if year != current_year:
            if current_year is not None:
                # Close the previous thematic area if it exists
                if current_goal_area is not None:
                    output += "        </ul>\n      </details>\n    </div>\n"
                    current_goal_area = None
                # Close the previous year block
                output += "  </details>\n"
            
            # Open a new year block
            goal_area_preview = ", ".join(df[df['year'] == year]['goal_area'].unique())
            output += f"  <details class='year-details'>\n    <summary class='year-summary'>{year} - {goal_area_preview}</summary>\n"
            current_year = year

        # Check if we are in a new thematic area
        if goal_area != current_goal_area:
            if current_goal_area is not None:
                # Close the previous thematic area <details> block if there is a previous thematic area
                output += "        </ul>\n      </details>\n    </div>\n"
            
            # Open a new thematic area block
            color = goal_area_colors.get(goal_area, "#ccc")
            total_goal_area = df[(df['goal_area'] == goal_area) & (df['year'] == year)].shape[0]
            output += f"    <div class='box'>\n"
            output += f"      <details>\n"
            output += f"        <summary style='background-color: {color}; color: white;'>{goal_area} ({str(total_goal_area)})</summary>\n"
            output += "        <ul>\n"
            current_goal_area = goal_area
        
        # Add the description as a list item
        output += f"          <li>{description}</li>\n"

    # Close any remaining open blocks
    if current_goal_area is not None:
        output += "        </ul>\n      </details>\n    </div>\n"
    if current_year is not None:
        output += "  </details>\n"

    output += "</div>"
    
    return output


def generate_programme_markdown(df):
    # Create an empty string to store the markdown content
    markdown_content = ""

    # Group the dataframe by 'PCR_NAME'
    grouped_df = df.groupby('PCR_NAME')

    for pcr_name, group in grouped_df:
        # Add the H3 title for each distinct PCR_NAME
        markdown_content += f"### {pcr_name}\n\n"

        for _, row in group.iterrows():
            intermediate_result_name = row['INTERMEDIATE_RESULT_NAME']
            ir_full_text = row['IR_FULL_TEXT']
            
            # Add collapsible section with bold title for INTERMEDIATE_RESULT_NAME and IR_FULL_TEXT inside
            markdown_content += f"<details class='year-details'>\n"
            if not pd.isnull(row['UTILIZED']):
                utilized = utils.format_number(row['UTILIZED'])
                markdown_content += f"<summary class='year-summary'><b>{intermediate_result_name} - Utilized ({str(utilized)}$)</b></summary>\n\n"
            else:
                markdown_content += f"<summary class='year-summary'><b>{intermediate_result_name}</b></summary>\n\n"
            markdown_content += f"{ir_full_text}\n"
            markdown_content += f"</details>\n\n"

    return markdown_content

def plot_sankey(df, source_node='goal_area', target_node='strategy_name', aggregation_method='utilized_sum'):
    """
    Plots a Sankey diagram with different aggregation methods.
    
    Parameters:
    - df: The dataframe containing the data.
    - source_node: The source node for the Sankey diagram.
    - target_node: The target node for the Sankey diagram.
    - aggregation_method: The method of aggregation to use. 
        Options are:
        - 'utilized_sum' : Sum of the 'expenditure' column grouped by source_node
        - 'activity_count': Unique count of 'activity_code' grouped by source_node
        - 'output_count'  : Unique count of 'output_code' grouped by source_node
    """
    # Filter the data and define the aggregation method
    if aggregation_method == 'utilized_sum':
        df = df.dropna(subset=['expenditure'])
        df_grouped = df.groupby([source_node, target_node]).agg({'expenditure': 'sum'}).reset_index()
        values = df_grouped['expenditure'].tolist()

    elif aggregation_method == 'activity_count':
        df_grouped = df.groupby([source_node, target_node]).agg({'activity_code': pd.Series.nunique}).reset_index()
        values = df_grouped['activity_code'].tolist()

    elif aggregation_method == 'output_count':
        df_grouped = df.groupby([source_node, target_node]).agg({'output_code': pd.Series.nunique}).reset_index()
        values = df_grouped['output_code'].tolist()

    else:
        print(f"Invalid aggregation method: {aggregation_method}")
        return

    # Calculate total for each source node and sort in descending order
    source_totals = df_grouped.groupby(source_node).sum().reset_index()
    source_totals = source_totals.sort_values(by=source_totals.columns[1], ascending=False)

    # Get the sorted list of source nodes based on the total
    sorted_source_nodes = source_totals[source_node].tolist()

    # Target nodes remain in the order they appear in the filtered data
    target_nodes = df_grouped[target_node].unique().tolist()

    # Combine sorted source nodes and target nodes to create the full set of labels
    all_labels = sorted_source_nodes + target_nodes

    # Map source nodes and target nodes to their respective positions in the all_labels list
    source_indices = df_grouped[source_node].apply(lambda x: all_labels.index(x)).tolist()
    target_indices = df_grouped[target_node].apply(lambda x: all_labels.index(x)).tolist()
    
    node_colors = [goal_area_colors.get(label, '#808080') for label in sorted_source_nodes]
    
    # Define link colors to match the color of the source node with transparency
    link_colors = [
        hex_to_rgba(goal_area_colors.get(label, '#808080'), alpha=0.2)
        for label in df_grouped[source_node]
    ]

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,  # Adjusted for better separation
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,  # Sorted labels by 'expenditure' for source nodes
            color=node_colors + ['#AAAAAA'] * len(target_nodes)  # Default color for result areas
        ),
        link=dict(
            source=source_indices,  # Adjusted indices based on sorting
            target=target_indices,  # Adjusted indices for targets
            value=values,
            color=link_colors,  # Transparent link colors, based on the source node
            hovertemplate='Source: %{source.label}<br>Target: %{target.label}<br>'
        )
    )])

    # Update the layout of the figure
    fig.update_layout(
        font_size=10,
    )

    return fig

# Function to wrap text to a maximum of 3 lines
def wrap_text(text, max_len=30, max_lines=3):
    words = text.split()
    lines, current_line = [], []
    
    for word in words:
        if len(' '.join(current_line + [word])) <= max_len:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
        if len(lines) == max_lines - 1:  # Limit to max_lines - 1 because we handle the last line outside the loop
            break
    
    if current_line:
        lines.append(' '.join(current_line))
    return '<br>'.join(lines[:max_lines])

# Function to generate the heatmap for each combination of goal_area and indicator_category
def generate_heatmap(df):
    color_continuous_scale=[
        (0.0, "white"),   # No Data
        (0.5, PALLETE_QUALITATIVE[1]),  # Not Achieved
        (0.75, PALLETE_QUALITATIVE[2]), # In Progress
        (1.0, PALLETE_QUALITATIVE[0])  # Achieved (UNICEF Blue)
    ]


    df['size'] = df.groupby(['indicator_code'], as_index=False).transform('size')

    # Filter for years >= 2018
    df = df[df['target_year'] >= 2018]

    mask = df['size'].gt(1)
    df = df.loc[mask].melt(
        id_vars=['country', 'indicator_code', 'indicator', 'indicator_unit', 'rating', 'target_year'],
        value_vars=['target_value', 'indicator_actual'],
    )
    df['variable'] = df['variable'].replace({'target_value': 'Target Value', 'indicator_actual': 'Observed Value'})

    df['rating_value'] = df['rating'].map(rating_mapping)

    df_pivot = df.query("variable == 'Target Value'").pivot(index=['indicator_code', 'indicator'], columns='target_year', values='rating_value').fillna(0)
    
    # Create a hover text array
    hover_text = []
    for i, indicator in enumerate(df_pivot.index):
        hover_text_row = []
        for j, year in enumerate(df_pivot.columns.get_level_values(0)):  # Corrected level reference
            text_indicator = wrap_text(indicator[1])  # Use the 'indicator' with text wrapping
            text_year = year 

            status = df[(df['target_year'] == year) & (df['indicator_code'] == indicator[0])]['rating']
            if not status.empty:
                status = status.values[0]
            else:
                status = 'No Data'

            hover_text_row.append(f"Indicator: {text_indicator}<br>Year: {year}<br>Status: {status}")
        hover_text.append(hover_text_row)

    # Create the heatmap using go.Heatmap to allow for borders
    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=df_pivot.columns.get_level_values(0),  # Correct column level
        y=[wrap_text(indicator[1]) for indicator in df_pivot.index],  # Display the indicator with text wrapping
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        colorscale=color_continuous_scale,
        showscale=False,  # Hide colorbar
        zmin=0,  # Min value for color mapping
        zmax=1,  # Max value for color mapping
        # Add borders
        xgap=2,  # Gap between cells along the x-axis
        ygap=5,  # Increase the gap between cells along the y-axis to prevent overlap
    ))

    # Update layout for aspect ratio, background color, and plot size
    aspect_ratio = df_pivot.shape[0] / df_pivot.shape[1]
    fig.update_layout(
        plot_bgcolor='white',  # Set the plot background color to white
        paper_bgcolor='white',  # Set the paper background color to white
        #width=1000,  # Set the width of the figure
        height=400 + 50 * df_pivot.shape[0],  # Increase height to add more space based on the number of rows
        xaxis=dict(
            showgrid=False,  # Disable x-axis grid
            tickvals=df_pivot.columns.get_level_values(0),  # Set tick values for the x-axis (years)
            ticktext=df_pivot.columns.get_level_values(0),  # Set tick labels for the x-axis (years)
            side="bottom"  # Place x-axis tick labels at the bottom
        ),
        yaxis=dict(
            showgrid=False,
            tickmode='array',
            tickvals=list(range(len(df_pivot.index))),
            ticktext=[wrap_text(indicator[1]) for indicator in df_pivot.index],
            ticks="outside",  # Place y-axis ticks outside
            ticklen=10,  # Increase tick length for readability
            automargin=True,  # Automatically adjust margins to prevent overlaps
        )
    )

    # Duplicate the x-axis for top tick labels
    fig.update_layout(
        xaxis2=dict(
            showgrid=False,
            tickvals=df_pivot.columns.get_level_values(0),  # Set tick values for the top x-axis
            ticktext=df_pivot.columns.get_level_values(0),  # Set tick labels for the top x-axis
            side="top",  # Place x-axis tick labels at the top
            matches='x'  # Ensure the top axis matches the main x-axis
        )
    )

    # Add border for cells
    fig.update_traces(
        xgap=2,  # Gap between cells along the x-axis
        ygap=5   # Increase the gap between cells along the y-axis to prevent overlap
    )

    return fig

def first_year(text):
    # Regular expression to find a 4-digit year like 2024
    match = re.search(r'(20\d{2})', text) 
    if match:
        return match.group(1)
    return None


def plot_cp(df: pd.DataFrame):
    df = df.groupby('cp')['value'].sum().reset_index()
    df['start_year'] = df['cp'].apply(first_year)

    # remove previous CP data and re-order year to ascending
    df.dropna(axis=0, inplace=True)
    df = df.sort_values(by='start_year', ascending=True).reset_index(drop=True)

    # re-order colours to match the treemap plot colors for each CP
    df_sorted = df.sort_values(by='value', ascending=False)
    color_map = {
        value: PALLETE_QUALITATIVE[i] for i, 
        value in enumerate(df_sorted['value'])}
    df['color'] = df['value'].map(color_map)

    fig = px.bar(
        data_frame=df,
        x="cp",
        y="value",
        color="cp",
        color_discrete_sequence= df['color'],
        hover_data={'color': False},
        labels={
            "cp": "",
            "value":"Funds Utilization in USD"
        }
    )
    fig.update_layout(
        showlegend=False,
    )
    return fig

def plot_ram_indicators(df: pd.DataFrame, x_value, rating_type):
    '''
    x_value [str]: year type you want for plot (eg. target_year, finalization_year) 
    rating_type [str]: type of assesment ('End-year assessment','End-Term','Mid-year assessment')
    
    '''
    column_names = df.columns[1:4].tolist()
    fig = px.bar(
        data_frame=df,
        x=x_value,
        y=column_names,
        color_discrete_sequence= PALLETE_QUALITATIVE,
        title=rating_type,

    )

    return fig

def plot_tags(df: pd.DataFrame, label, y_title):
    column_names = df.columns[3:].tolist()  # Exclude the first three columns
    fig = go.Figure()

    for i, column in enumerate(column_names):
        # Add line traces for Tags
        fig.add_trace(go.Scatter(
            x=df['cp_short_text'],
            y=df[column],
            name=column,
            mode='lines+markers',
            line=dict(color=PALLETE_QUALITATIVE[i], width=4),
            marker=dict(size=10),  
            # marker=dict(size=10, line=dict(width=2, color='white')),  # Outline for markers
            hovertemplate=f'{column}: %{{y:.1f}}<extra></extra>',  # Format y to 1 significant figure
            showlegend=True
        ))

    # Update layout for y-axes and legend
    fig.update_layout(
        xaxis_title='',
        yaxis_title=y_title,
        # yaxis=dict(range=[-5, 105]),  # Set the y-axis range to [0, 100]
        xaxis=dict(showgrid=False),  # Remove the x-axis lines
        legend_title_text=label,
        showlegend=True 
    )

    return fig

def plot_prog_approach_bar(df: pd.DataFrame):
    
    df.sort_values(['country_programme', 'total'], ascending=[True, False], ignore_index=True, inplace=True)
    # Create subplots
    fig = sp.make_subplots(rows=1, cols=2)
    unique_cp = df['country_programme'].unique().tolist()
    
    years1 = [str(year) for year in range(2016, 2024)]
    years2 = [str(year) for year in range(2021, 2025)]
    
    # Function to add stacked bar chart for a given country programme (only 2 cp work)
    def add_stacked_bar_chart(country_programme, years):
        df_filtered = df[df['country_programme'] == country_programme]   
        for i, strategy in enumerate(df_filtered['SP Change Strategies / GICs / Related Outputs']):
            wr_strategy = utils.wrap_text_with_br(strategy,width=30)
            values = df_filtered[df_filtered['SP Change Strategies / GICs / Related Outputs'] == strategy][years].values.flatten()
            fig.add_trace(go.Bar(
                x=years, 
                y=values,
                name=f"{country_programme} - {wr_strategy}",  # Prefix strategy name with country programme
                marker=dict(color=PALLETE_QUALITATIVE[i]), 
                hoverinfo='y',
                ), 
                row=1, col=(1 if country_programme == unique_cp[0] else 2
                            )
                )
        
    # Add charts for cp1 and cp2
    add_stacked_bar_chart(unique_cp[0], years1)
    add_stacked_bar_chart(unique_cp[1], years2)

    # Update layout
    fig.update_layout(
        yaxis_title='Funds Utilization in USD',
        barmode='stack',
        legend_title_text='Strategies',
        showlegend=True,
    )
    # Update x-axis titles for each subplot
    fig.update_xaxes(title_text=unique_cp[0], row=1, col=1)
    fig.update_xaxes(title_text=unique_cp[1], row=1, col=2)
    
    return fig


def plot_prog_approach_sankey(df: pd.DataFrame, title):
    to_keep = ['SP Change Strategies / GICs / Related Outputs', 'level', 'total']
    df = df.reindex(to_keep, axis=1)

    # Create lists to hold the source/target names and link values 
    source = []
    target = []
    values = []

    #indices to assign source/target and colors
    index_strategy = 0
    index_GIC = 0
    index_output = 0
    color_index = -1
    source_color_map = {}
    link_color_list = []

    # Iterate through the DataFrame to build the source, target, and values lists
    for i, row in df.iterrows():
        if row['level'] == 'Strategy':
            color_index += 1
            index_strategy = i
            source_color_map[row['SP Change Strategies / GICs / Related Outputs']] = PALLETE_QUALITATIVE[color_index]
        
        if row['level'] == 'GIC':
            index_GIC = i
            source.append(df.loc[index_strategy, 'SP Change Strategies / GICs / Related Outputs'])
            target.append(df.loc[index_GIC, 'SP Change Strategies / GICs / Related Outputs'])
            values.append(row['total'])
            link_color_list.append(PALLETE_QUALITATIVE[color_index])
            source_color_map[row['SP Change Strategies / GICs / Related Outputs']] = PALLETE_QUALITATIVE[color_index]

        if row['level'] == 'Output':
            index_output = i
            source.append(df.loc[index_GIC, 'SP Change Strategies / GICs / Related Outputs'])
            target.append(df.loc[index_output, 'SP Change Strategies / GICs / Related Outputs']) 
            values.append(row['total'])
            link_color_list.append(PALLETE_QUALITATIVE[color_index])
            source_color_map[row['SP Change Strategies / GICs / Related Outputs']] = '#808080'
            # place the sources, targets, values, and link_colors in a dictionary to extract indices for sources/targets 

    data = {'source': source,
            'target': target,
            'values': values,
            'link_colors': link_color_list,
            }

    df_plot = pd.DataFrame(data)

    # find all unique node names in sources and targets
    all_nodes = list(pd.unique(df_plot[['source', 'target']].values.ravel('K')))

    # map node names to indices for Sankey plot
    mapping = {node: idx for idx, node in enumerate(all_nodes)}

    df_plot['source_idx'] = df_plot['source'].map(mapping)
    df_plot['target_idx'] = df_plot['target'].map(mapping)

    # df_sankey_plot.sort_values(['source_idx'], ascending=[True], ignore_index=True, inplace=True)

    # get all the node colors
    node_colors = [source_color_map.get(label, '#808080') for label in all_nodes]

    # Function to add transparency to the color
    def add_transparency(color_hex, alpha=0.5):
        # Convert HEX to RGB
        color_rgb = [int(color_hex[i:i+2], 16) for i in (1, 3, 5)]
        return f'rgba({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}, {alpha})'

    # Desired transparency level
    alpha_value = 0.5  # Adjust this value as needed (0 to 1)

    # Create a new list with RGBA colors
    transparent_link_color_list = [add_transparency(color, alpha_value) for color in link_color_list]
    
    # Apply text wrapping to all labels
    wrapped_nodes = [utils.wrap_text_with_br(label, width = 45) for label in all_nodes]

    # Create the Sankey diagram
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=8,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=wrapped_nodes,
            color = node_colors
        ),
        link=dict(
            source=df_plot['source_idx'],
            target=df_plot['target_idx'],
            value=df_plot['values'],
            color=transparent_link_color_list
        )
    ))
    fig.update_layout(title_text=title, font_size=10)

    return fig

def plot_staff_turnover(df: pd.DataFrame):
    column_names = ['turnover', 'local_turnover', 'foreign_turnover']
    fig = px.line(
        df, 
        x='year', 
        y=column_names, 
        color_discrete_sequence= PALLETE_QUALITATIVE, 
        markers=True,
        labels={'variable': 'Staff'}
        )
    
    fig.update_traces(line=dict(width=4), marker=dict(size=10))
    # Update the legend names using update_traces
    fig.for_each_trace(lambda trace: trace.update(name={
        'turnover': 'Total Staff',
        'local_turnover': 'Local Staff',
        'foreign_turnover': 'Foreign Staff'
    }[trace.name]))
    fig.update_layout(
        xaxis_title='Years',
        yaxis_title='Staff Turnover (%)'
        )

    return fig

def plot_staff_type(df: pd.DataFrame, column_names, label):
    fig = px.bar(
        data_frame=df,
        x='year',
        y=column_names,
        color_discrete_sequence= PALLETE_QUALITATIVE,
        labels={'variable': label}
    )

    fig.update_layout(xaxis_title='Years',
                      yaxis_title='Number of Staff' )

    return fig

def plot_staff_thematic_area(df: pd.DataFrame):
    fig = px.bar(
        data_frame=df,
        x='year',
        y='staff_proportion',
        color='thematic_area',
        color_discrete_sequence= PALLETE_QUALITATIVE,
        labels={'thematic_area': 'Thematic Area'},
        hover_data={'staff_proportion': ':.1f'}  # Format hover to 1 significant figure
    )

    fig.update_layout(xaxis_title='Years',
                      yaxis_title='Staff (%)' )

    return fig

def plot_donor_funds(df: pd.DataFrame):
    unique_cp = df['cp'].unique().tolist()

    # Create a 3-row, 1-column subplot layout
    row_no = 3
    fig = sp.make_subplots(rows=row_no, cols=1, subplot_titles=unique_cp)

    # Create a color map for donor_level2
    unique_donors = df['donor_level2'].unique()
    unique_donors.sort()
    color_map = {donor: color for donor, color in zip(unique_donors, PALLETE_QUALITATIVE)}

    # Loop through the CP values to create each subplot
    for i, cp in enumerate(unique_cp):
        # Filter the DataFrame for the current CP value
        df_cp = df[df['cp'] == cp].copy()
        df_cp.sort_values(by=['funds_type','donor_level2', 'grand_total'], ascending=[True, False, False], inplace=True)
        # Assign colors based on the donor_level2 values using the color map
        bar_colors = [color_map[donor] for donor in df_cp['donor_level2']]
        # Create a bar chart for the current CP value
        fig.add_trace(go.Bar(
            y=df_cp['funds_type'],
            x=df_cp['grand_total'],
            marker=dict(color=bar_colors),  # Use the mapped colors
            hoverinfo='text',
            hovertext=df_cp.apply(
                lambda row: f"Donor: {row['donor']}<br>Donor Class (Level 2): {row['donor_level2']}<br>Total USD: {int(row['grand_total'])}", axis=1),
            orientation='h',
            showlegend=False  # Omit this trace from the legend        
        ), row=i + 1, col=1
        )

    # Add dummy traces for the legend
    for donor, color in color_map.items():
        fig.add_trace(go.Bar(
            x=[None],  # No actual data, just for legend
            y=[None],
            name=donor,
            marker_color=color,
            showlegend=True
        ))

    # Update layout
    fig.update_layout(height=900, showlegend=True, legend=legend_style, legend_title_text="")

    # Set y-axis range for alignment across subplots
    # Replace 'min_value' and 'max_value' with appropriate limits based on your data
    min_value = 0 
    max_value = df.groupby(['cp', 'funds_type'], as_index=False)['grand_total'].sum()['grand_total'].max() 
    for i in range(1, 4):  # Assuming you have 3 rows
        fig.update_xaxes(range=[min_value, max_value], row=i, col=1)

    fig.update_xaxes(title_text='Allocated Funds in USD', row=row_no, col=1)
  
    return fig

def plot_SDG_funds(df: pd.DataFrame):
    unique_cp = df['cp'].unique().tolist()

    # Create a 2-row, 1-column subplot layout, using last 2 CP
    row_no = 2
    fig = sp.make_subplots(rows=row_no, cols=1, subplot_titles=unique_cp[-2:])
    
    # Create a mapping from goal_area_code to goal_area for hover text
    goal_area_map = df.set_index('goal_area_code')['goal_area'].to_dict()
    # Loop through the CP to create each subplot
    for i, cp in enumerate(unique_cp[-2:]):
        # Filter the DataFrame for the current CP value
        df_cp = df[df['cp'] == cp].copy()
        df_cp.sort_values(by=['funds_type', 'goal_area_code'], ascending=[True, True], inplace=True)
        # Assign colors based on the goal area codes using the color map
        bar_colors = [SDG_color_map[code] for code in df_cp['goal_area_code']]
                
        # Create a bar chart for the current CP value
        fig.add_trace(go.Bar(
            y=df_cp['funds_type'],
            x=df_cp['funds'],
            marker=dict(color=bar_colors),  # Use the mapped colors
            hoverinfo='text',
            hovertext=df_cp.apply(
                lambda row: f"SDG Area: {row['goal_area']}<br>Total USD: {int(row['funds'])}", axis=1),
            orientation='h',
            showlegend=False  # Omit this trace from the legend        
        ), row=i + 1, col=1)

    # Add dummy traces for the legend
    for area_code, color in SDG_color_map.items():
        area_name = goal_area_map.get(area_code, None)  # Get goal_area name or fallback to None
        if area_name != None:
            fig.add_trace(go.Bar(
                x=[None],  # No actual data, just for legend
                y=[None],
                name=area_name,
                marker_color=color,
                showlegend=True
            ))

    # Update layout
    fig.update_layout(height=900, showlegend=True, legend=legend_style, legend_title_text="")

    # Set y-axis range for alignment across subplots
    min_value = 0 
    max_value = df.groupby(['cp', 'funds_type'], as_index=False)['funds'].sum()['funds'].max() 
    for i in range(1, 3):  # Assuming you have 2 rows
        fig.update_xaxes(range=[min_value, max_value], row=i, col=1)

    fig.update_xaxes(title_text='Expenses in USD', row=row_no, col=1)
    
    return fig


def plot_SDG_agencies(df: pd.DataFrame):
    SDG_map = {SDG_goals[key]: SDG_color_map[key] for key in SDG_goals if key in SDG_color_map}
    sdg_order = list(SDG_goals.values())
    # Set the SDG column as a categorical type with the specified order
    df['SDG'] = pd.Categorical(df['SDG'], categories=sdg_order, ordered=True)
    df.sort_values(by=['SDG'], inplace=True, ascending=[True])
   
    # Create the horizontal bar chart
    fig = px.bar(
        df,
        y='agencies',
        x='total_expenditure',
        color='SDG',
        color_discrete_map=SDG_map,
        hover_data={'agencies': False},
        orientation='h'
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        legend=legend_style,
        legend_title_text="",
        xaxis_title='Expenses in USD',
        yaxis_title=''
    )
    
    # fig.write_html(COUNTRY + '_agencies.html')
    return fig


def plot_agencies_count(df: pd.DataFrame):
    fig = px.bar(df, y='agency_abbreviations', x='count', color='type', 
                 orientation='h',hover_data={'agency_abbreviations': False}, 
                 color_discrete_sequence=PALLETE_QUALITATIVE,
                )
    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        legend=legend_style,
        legend_title_text="",
        xaxis_title='Count',
        yaxis_title=''
    )
    return fig


def generate_markdown_by_country(df):
    markdown_output = ""
    
    # Get a sorted list of unique years for the country
    years = sorted(df['year'].unique())
    
    # Iterate over each year
    for year in years:
        # Add the year as an H3 heading
        markdown_output += f"### {year}\n"
        
        # Filter data for the current year and maintain order
        year_data = df[df['year'] == year]
        
        # Track the current goal area to avoid repeating it
        current_goal_area = None
        
        # Iterate over each row in the year_data DataFrame while maintaining order
        for _, row in year_data.iterrows():
            goal_area = row['goal_area']
            need = row['need']
            
            # If the goal area changes, add a new bullet point for the goal area
            if goal_area != current_goal_area:
                markdown_output += f"- **{goal_area}**:\n"
                current_goal_area = goal_area
            
            # Add the need as a nested bullet point under the goal area
            markdown_output += f"  - {need}\n"
    
    return markdown_output


def plot_innovation(df_activities, metric='expenditures'):
    # Filter for the specific  innovation-related activities
    innovation_activities = df_activities[
        (df_activities['generic_intervention_name'].isin([
            'Fostering innovation and use of new technologies', 
            'Innovation', 
            'Digital Transformation'
        ]))
    ]
    
    # Check if there is any data for innovation activities
    if innovation_activities.empty:
        return None
    
    # Choose the appropriate metric to calculate
    if metric == 'expenditures':
        # Group by year and sum the 'value' to get resource allocation over time for innovation
        resources_over_time = innovation_activities.groupby('year')['value'].sum().reset_index()

        # Get total expenditures by year (for percentage calculation)
        total_resources_over_time = df_activities.groupby('year')['value'].sum().reset_index()

        # Merge to calculate the percentage of innovation expenditures
        merged_data = pd.merge(resources_over_time, total_resources_over_time, on='year', suffixes=('_innovation', '_total'))
        merged_data['percentage_innovation'] = (merged_data['value_innovation'] / merged_data['value_total']) * 100
        merged_data['year'] = merged_data['year'].astype(str)

        # Titles and labels
        yaxis_title = "Expenditures (USD)"
        secondary_yaxis_title = "% of Expenditures for Innovation"
        primary_hover_template = '%{y:$,.2f}'
        secondary_hover_template = '%{y:.2f}%'

    elif metric == 'activities':
        # Group by year and count the number of innovation-related activities
        activity_count_over_time = innovation_activities.groupby('year').size().reset_index(name='activity_count')

        # Get total activities by year (for percentage calculation)
        total_activity_count_over_time = df_activities.groupby('year').size().reset_index(name='total_activity_count')

        # Merge to calculate the percentage of innovation activities
        merged_data = pd.merge(activity_count_over_time, total_activity_count_over_time, on='year')
        merged_data['percentage_innovation'] = (merged_data['activity_count'] / merged_data['total_activity_count']) * 100

        # Titles and labels
        yaxis_title = "Count of Activities"
        secondary_yaxis_title = "% of Innovation Activities"
        primary_hover_template = '%{y:,}'  # For activity counts
        secondary_hover_template = '%{y:.2f}%'

    else:
        return "Invalid metric. Please choose 'expenditures' or 'activities'."

    # Create the Plotly figure
    fig = go.Figure()

    # Bar chart for the chosen metric (either expenditures or activity count)
    fig.add_trace(go.Bar(
        x=merged_data['year'], 
        y=merged_data[merged_data.columns[1]],  # Either 'value_innovation' or 'activity_count'
        name=f"{yaxis_title}",
        marker_color=PALLETE_QUALITATIVE[0],
        hovertemplate=primary_hover_template
    ))

    # Line chart for percentage of the chosen metric (either expenditures or activity count)
    fig.add_trace(go.Scatter(
        x=merged_data['year'], 
        y=merged_data['percentage_innovation'],
        mode='lines+markers',
        name=f"{secondary_yaxis_title}",
        line=dict(color=PALLETE_QUALITATIVE[1], width=2),
        marker=dict(color=PALLETE_QUALITATIVE[1]),
        hovertemplate=secondary_hover_template,
        yaxis="y2"  # Use the secondary y-axis
    ))

    # Update the layout with secondary y-axis
    fig.update_layout(
        xaxis_title="Year",
        yaxis=dict(
            title=yaxis_title,
            tickformat=",",  # Use commas for large numbers
        ),
        yaxis2=dict(
            title=secondary_yaxis_title,
            overlaying='y',
            side='right',
            tickformat=".2f%%"  # Show percentages on the secondary y-axis
        ),
        legend=legend_style,
        legend_title_text="",
        barmode='group'
    )

    # Show the plot
    return fig


def plot_stacked_bar_chart(df, stack_by='goal_area', count_of='output_code', aggregation_type='nunique'):

    # Step 1: Group the data by year and stack_by, then count distinct activity codes
    grouped_df = df.groupby(['year', stack_by]).agg(distinct_activity_codes=(count_of, aggregation_type)).reset_index()

    pivot_df = grouped_df.pivot_table(index='year', columns=stack_by, values='distinct_activity_codes', aggfunc='sum').fillna(0)

    # Create an empty figure
    fig = go.Figure()

    # Add a trace for each goal area, with corresponding colors from the goal_area_colors dictionary
    for stack in pivot_df.columns:
        fig.add_trace(go.Bar(
            x=pivot_df.index,  # Years
            y=pivot_df[stack],  # Activity counts for this goal area
            name=utils.beautify_labels(stack),  # Legend label
            marker_color=goal_area_colors.get(stack, '#333333')  # Use the color from the dictionary, default to grey
        ))

    # Step 5: Customize the layout and move the legend to the bottom
    fig.update_layout(
        barmode='stack',
        xaxis_title='Year', 
        yaxis_title=f'{utils.beautify_labels(count_of)} by {utils.beautify_labels(stack_by)}',
        legend=legend_style,
        legend_title_text="",
        margin=dict(b=50)
    )
    
    return fig


def add_drop_dow(figures, selection):
    """
    Creates a Plotly figure with a dropdown to switch between different years or categories,
    preserving the layout from the original figures.
    
    Parameters:
    - figures (dict): A dictionary where keys are years or categories (e.g., '2020', '2021')
                      and values are Plotly `fig` objects to display.
    - selection (list): A list of keys from `figures` to be used for dropdown options.

    Returns:
    - fig: A Plotly `fig` object with a dropdown to switch between the input figures.
    """
    # Set the initial figure to the first year (displayed by default)
    initial_year = selection[0]

    # Create the master figure
    fig = go.Figure()

    # Add traces for all years and set visibility appropriately
    total_traces = 0  # Counter for the total number of traces added
    trace_counts = {}  # Track the number of traces per year

    for year in selection:
        new_fig = figures[year]
        if new_fig is not None:
            trace_counts[year] = len(new_fig.data)  # Store trace count for each year
            for trace in new_fig.data:
                fig.add_trace(trace)
                fig.data[-1].visible = (year == initial_year)  # Only show initial year
            total_traces += len(new_fig.data)  # Update total trace count

    # Build visibility arrays for each dropdown option
    visibility_settings = []
    for year in selection:
        visibility = [False] * total_traces  # Start with all traces invisible
        start_index = sum(trace_counts[y] for y in selection if y < year)  # Calculate where the current year's traces start
        end_index = start_index + trace_counts[year]  # End index for current year's traces

        # Set current year's traces to visible
        for i in range(start_index, end_index):
            visibility[i] = True

        # Append the visibility setting for the current year
        visibility_settings.append(visibility)

    # Create dropdown buttons with visibility settings
    dropdown_buttons = [
        {
            'method': 'restyle',
            'label': str(year),
            'args': [{'visible': visibility_settings[i]}],
        }
        for i, year in enumerate(selection)
    ]

    # Copy layout from the first figure in the dictionary
    fig.update_layout(figures[initial_year].layout)

    # Update layout with the dropdown menu in the top-left corner
    fig.update_layout(
        legend=legend_style,
        legend_title_text="",
        updatemenus=[
            {
                'buttons': dropdown_buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0,  # Position dropdown on the left
                'y': 1.15,  # Position dropdown slightly above the plot area
                'xanchor': 'left',
                'yanchor': 'top'
            }
        ]
    )

    return fig





def plot_timeline_description_bar_chart(df_timeline, title_col='innovation_title', text_col='innovation_text'):
    # Count the number of innovations per year
    df_timeline['count'] = 1  # Each innovation counts as 1 unit
    

    df_timeline["Description"] = df_timeline[text_col].apply(
        lambda x: utils.wrap_text_with_br(x, width=70)
    )

    # Create the bar chart using Plotly Express
    fig = px.bar(
        df_timeline, 
        x='year', 
        y='count', 
        hover_name=title_col,  # Hover will show innovation title as the hover name
        hover_data={'year': True, 'Description': True, 'count': False},  # Show 'year' and 'innovation_text' in hover data
        color_discrete_sequence=[COLOR_PRIMARY]  # Use the same color for all innovations
    )
    
    fig.update_layout(
        barmode='stack',  # Stack bars
        xaxis_title="Year",
        yaxis_title="Innovations",
        showlegend=False,  # Remove legend if you don't need it
        hoverlabel=dict(namelength=-1), 
    )

    return fig  


def generate_timeline_markdown(df, pargagph_title='year', itermediate_title='innovation_title', description='innovation_text'):
    # Create an empty string to store the markdown content
    markdown_content = ""

    # Group the dataframe by pargagph_title
    grouped_df = df.groupby(pargagph_title)

    for pargagph, group in grouped_df:
        # Make the year collapsible using Quarto callout
        markdown_content += f"::: {{.callout-note collapse='true' icon=false appearance='minimal'}}\n"
        markdown_content += f"### {pargagph}\n\n"

        # Iterate over each row in the group for the given year
        for _, row in group.iterrows():
            itermediate_title_name = row[itermediate_title]
            description_text = row[description]
            
            # Nested collapsible callout for each innovation_title
            markdown_content += f"::: {{.callout-note collapse='true' icon=false appearance='minimal'}}\n"
            markdown_content += f"#### {itermediate_title_name}\n\n"
            markdown_content += f"{description_text}\n"
            markdown_content += f":::\n\n"
        
        # Close the year callout
        markdown_content += f":::\n\n"

    return markdown_content


def generate_collapsable_markdown(df, pargagph_title='goal_area', description='recommendation'):
    # Create an empty string to store the markdown content
    markdown_content = ""

    # Group the dataframe by pargagph_title and aggregate unique recommendations
    grouped_df = df.groupby(pargagph_title, as_index=False).agg({description: 'unique'}).copy()

    # Update pargagph_title to include the length of recommendations and format recommendations as a markdown list
    grouped_df[pargagph_title] = grouped_df.apply(lambda row: f"{row[pargagph_title]} ({len(row[description])})", axis=1)
    grouped_df[description] = grouped_df[description].apply(lambda recs: '\n'.join([f"- {rec}" for rec in recs]))

    # Generate the markdown content
    for index, row in grouped_df.iterrows():
        pargagph = row[pargagph_title]
        description_text = row[description]
        
        # Construct the collapsible markdown content
        markdown_content += f"::: {{.callout-note collapse='true' icon=false appearance='minimal'}}\n"
        markdown_content += f"#### {pargagph}\n\n"
        markdown_content += f"{description_text}\n"
        markdown_content += ":::\n\n"

    return markdown_content


def plot_cp_expenditure_overview(df, wbs_level='activity_name', group_by='goal_area'):

    # Step 1: Calculate percentage of distinct activities per group_by within the selected cp
    activity_counts = df.groupby(group_by)[wbs_level].nunique().reset_index(name='distinct_activities')
    total_activities_by_cp = activity_counts['distinct_activities'].sum()
    activity_counts['activity_percentage'] = (activity_counts['distinct_activities'] / total_activities_by_cp) * 100

    # Step 2: Calculate percentage of expenditures per group_by within the selected cp
    expenditure_sums = df.groupby(group_by)['expenditure'].sum().reset_index(name='total_expenditure')
    total_expenditure_by_cp = expenditure_sums['total_expenditure'].sum()
    expenditure_sums['expenditure_percentage'] = (expenditure_sums['total_expenditure'] / total_expenditure_by_cp) * 100

    # Step 3: Merge activity percentages and expenditure percentages
    merged_data = pd.merge(
        activity_counts[[group_by, 'activity_percentage']],
        expenditure_sums[[group_by, 'expenditure_percentage']],
        on=group_by
    )

    # Step 4: Sort by expenditure percentage in descending order
    merged_data = merged_data.sort_values(by='expenditure_percentage', ascending=True)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        subplot_titles=(r"% of Expenditures", r"% of Activities")
    )

    # Expenditure Percentage Plot
    fig.add_trace(go.Bar(
        y=merged_data[group_by],
        x=merged_data['expenditure_percentage'],
        name='Expenditure %',
        orientation='h',
        marker=dict(
            color=[goal_area_colors.get(ga, "#9E9E9E") for ga in merged_data[group_by]]
        ),
        hovertemplate=r'%{x:.2f}% of expenditures'
    ), row=1, col=1)

    # Activity Percentage Plot
    fig.add_trace(go.Bar(
        y=merged_data[group_by],
        x=merged_data['activity_percentage'],
        name='Activity %',
        orientation='h',
        marker=dict(
            color=[goal_area_colors.get(ga, "#9E9E9E") for ga in merged_data[group_by]],
            opacity=0.8  # Slightly transparent to visually separate
        ),
        hovertemplate=r'%{x:.2f}% of activities'
    ), row=1, col=2)

    # Update layout
    fig.update_layout(
        xaxis_title="%",
        xaxis2_title="%",
        showlegend=False
    )
    return fig


def create_bubble_chart(df, group_by='goal_area', x_axis='expenditure', y_axis='activity_name', frame='year'):
    """
    Generate a bubble chart for expenditure and activity analysis.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - group_by (str): Column to group data by, either 'goal_area' or 'strategy'.
    - x_axis (str): Column to sum for the X-axis, typically 'expenditure'.
    - y_axis (str): Column to count unique values for the Y-axis, typically 'activity_name'.
    
    Returns:
    - fig: A Plotly figure representing the bubble chart.
    """
    
    # Aggregate data
    grouping_by = [group_by]
    if frame is not None:
        grouping_by.append(frame)
    grouped = df.groupby(grouping_by).agg(
        total_expenditure=(x_axis, 'sum'),
        unique_activities=(y_axis, 'nunique'),
    ).reset_index()

    grouped['mean_expenditure_per_activity'] = grouped['total_expenditure'] / grouped['unique_activities']
    # Determine color column based on grouping

    # Generate the bubble chart with custom hover data
    fig = px.scatter(
        grouped,
        x='total_expenditure',
        y='unique_activities',
        size='mean_expenditure_per_activity',
        color = group_by,
        animation_frame=frame, 
        animation_group=group_by,
        color_discrete_map = goal_area_colors,
        labels={
            'total_expenditure': 'Total Expenditure',
            'unique_activities': 'Number of Unique Activities',
            'mean_expenditure_per_activity': 'Mean Expenditure per Activity'
        },

        size_max=50,
        range_x=[-1, grouped['total_expenditure'].max()*1.1], 
        range_y=[-1,grouped['unique_activities'].max()*1.1],
        hover_data={
            'total_expenditure': ':$,.0f',  # Format as currency with no decimals
            'unique_activities': True,       # Show as is
            'mean_expenditure_per_activity': ':$,.0f',  # Format as currency with no decimals
            
        }
    )


    # Update layout to explicitly set axis titles and position the legend at the bottom
    fig.update_layout(
        xaxis_title="Total Expenditure",
        yaxis_title="Number of Unique Activities",
        legend=legend_style,
        legend_title_text="",
        showlegend=True,
        sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "center",
        "x": 0.5,
        "y": -0.22,  # Position slider above the chart
        "len": 0.9,
        "currentvalue": {"prefix": frame + ": "}
    }]
    )
    fig["layout"].pop("updatemenus") # optional, drop animation buttons

    return fig



def create_stacked_area_chart(df, group_by, value_column, operation):
    """
    Generate a stacked area chart based on the specified grouping, value, and aggregation,
    using the predefined UNICEF goal area colors.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - group_by (str): Column to group by, which will define the stacked areas (e.g., 'goal_area' or 'strategy').
    - value_column (str): Column containing the values to aggregate (e.g., 'expenditure' or 'activity_name').
    - operation (str): Aggregation operation to apply, either 'sum' or 'nunique'.
    
    Returns:
    - fig: A Plotly figure representing the stacked area chart.
    """
    # Validate operation input
    if operation not in ["sum", "nunique"]:
        raise ValueError("Operation should be either 'sum' or 'nunique'")
    
    # Choose aggregation function based on the specified operation
    agg_func = "sum" if operation == "sum" else "nunique"

    # Aggregate data by year and the specified group_by column
    grouped = df.groupby(['year', group_by]).agg(
        aggregated_value=(value_column, agg_func)
    ).reset_index()

    # Pivot the data to have years as rows and group_by categories as columns for the area chart
    pivoted = grouped.pivot(index='year', columns=group_by, values='aggregated_value').fillna(0)

    # Define the color map
    color_map = goal_area_colors 
    # Create a stacked area chart using Plotly
    fig = px.area(
        pivoted,
        x=pivoted.index,  # Year on the x-axis
        y=pivoted.columns, # Stack by each category in the group_by column
        labels={'value': value_column.capitalize(), 'year': 'Year'},
        color_discrete_map=color_map  # Apply the UNICEF color scheme
    )

    # Update layout to set legend at the bottom
    fig.update_layout(
        legend=legend_style,
        legend_title_text="",
        showlegend=True,
        xaxis_title="Year",
        yaxis_title=value_column.capitalize(),
    )

    return fig


def plot_hr_count_against_utilized_by_goal_area(tmp_df: pd.DataFrame):
    fig = px.scatter(
        tmp_df, 
        x="utilized", 
        y="hr_count", 
        animation_frame="year", 
        animation_group="goal_area",
        size="mean_utilized", 
        color="goal_area", 
        color_discrete_map = goal_area_colors,
        hover_name="goal_area",
        #log_x=True, 
        size_max=55, 
        #range_x=[0,100000000], 
        #range_y=[0,100],
        labels={
            "utilized": "Programme Funds Utilization",
            "hr_count": "HR count",
            "goal_area": "Goal Area",
            "mean_utilized": "Mean Utilization per HR unit",
            'year': 'Year'
        }
    )

    fig["layout"].pop("updatemenus") # optional, drop animation buttons
    fig.update_layout(

        legend=legend_style,
        legend_title_text="",
        showlegend=True,
        sliders=[{
        "active": 0,
        "yanchor": "top",
        "xanchor": "center",
        "x": 0.5,
        "y": -0.22,  # Position slider above the chart
        "len": 0.9,
    }]
    )
    return fig


def plot_partner_count_by_country_year(tmp_df: pd.DataFrame):
    fig = px.bar(
        tmp_df,
        x='year',
        y='implementing_partner',
        labels={'year': 'FR Start Year', 'implementing_partner': 'Number of Implementing Partners'},
        color_discrete_sequence=PALLETE_QUALITATIVE,
    )
    return fig


def plot_partner_count_new_to_past(df: pd.DataFrame):
    fig = px.bar(
        df,
        x='year',
        y=["Existing partner", 'New partner'],
        labels={'value': 'Number of implementing partners', 'variable': '', 'year': 'FR start year'},
        color_discrete_sequence=PALLETE_QUALITATIVE,
    )

    return fig


def plot_partner_fr_consumed_by_year(df: pd.DataFrame, labels: list = ['Below 100K', '100K to 1M', 'Above 1M']):
    # Define the color mapping
    color_map = {label: PALLETE_QUALITATIVE[i] for i, label in enumerate(labels)}

    fig = go.Figure()
    for label in labels:
            try:
                fig.add_trace(go.Bar(
                    x=df['year'],
                    y=df['fr_amount'][label],
                    name=label,
                    text=df['implementing_partner'][label],
                    textposition='auto',
                    marker_color=color_map[label],
                    yaxis='y'
                ))
            except:
                continue

    fig.update_layout(
            barmode='stack',
            xaxis_title='FR Start Year',
            yaxis_title='Total FR Amount',
            legend_title='Partner Yearly FR Amount',
        )
        
    return fig

def plot_partner_count_by_gic(df: pd.DataFrame):
    color_map = partner_sub_types_colors

    df = df.pivot(index=['generic_intervention_name'], columns='partner_type', values='implementing_partner')

    fig = px.bar(
                df,
                x = df.columns,
                y = df.index,
                labels={'partner_type': '', 'generic_intervention_name': 'GIC', 'value': 'Number of Implementing Partners'},
                color_discrete_sequence= PALLETE_QUALITATIVE,
                color_discrete_map=color_map,
                orientation='h'
            )


    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        legend=legend_style,
        legend_title_text=""
        )
    
    return fig
    


# Function to process data for plotting
def process_posts_for_plotting(posts):
    # Explode the 'label' column
    posts_exploded = posts.explode('label')

    # Group by year and label to get counts
    goal_area_counts = posts_exploded.groupby(['year', 'label']).size().reset_index(name='count')

    # Calculate total posts per year
    total_posts_per_year = posts_exploded.groupby('year').size().reset_index(name='total_posts')

    # Merge counts with total posts per year
    goal_area_counts = goal_area_counts.merge(total_posts_per_year, on='year')

    # Calculate proportion
    goal_area_counts['proportion'] = goal_area_counts['count'] / goal_area_counts['total_posts']

    # Pivot the table for plotting
    pivot_table = goal_area_counts.pivot(index='year', columns='label', values='proportion').fillna(0)

    # Reset index for plotting
    pivot_table = pivot_table.reset_index()

    # Melt the DataFrame for plotting
    melted_df = pivot_table.melt(id_vars='year', var_name='Goal Area', value_name='Proportion')

    return melted_df


def plot_sm_funds_bubble(df:pd.DataFrame, y, y_title):
    fig = px.scatter(df, x='year', y=y, size='share', color='goal_area', 
                    color_discrete_map=goal_area_colors, 
                    hover_name='goal_area', opacity=0.6, size_max=55,
                    labels={'share': 'Social Media Posts (share)'},
                    hover_data={'share': ':.3f', 'goal_area': False}
                    )
    fig.update_layout(height=900, showlegend=True, legend=legend_style, legend_title_text="",
                    xaxis_title='Years', yaxis_title=y_title
                    )
    return fig