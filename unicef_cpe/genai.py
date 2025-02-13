"""
Functions for working with generative models via OpenAI API.
"""
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def prompt(
    user_message: str,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    api_type: str = "openai",  # Default is OpenAI, switch to "ollama" for Ollama
    seed: int = 42  # Seed for reproducibility
) -> str:
    
    if api_type == "openai":
        # LangChain OpenAI API wrapper with seed
        openai_llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            seed=seed  # Using seed for reproducibility
        )
        messages = [
            ("system", system_message),
            ("human", user_message),
        ]
        response = openai_llm.invoke(messages)
        content = response.content
    
    elif api_type == "ollama":
        # LangChain Ollama API wrapper with seed
        ollama_llm = ChatOllama(
            model=model,
            temperature=temperature,
            seed=seed  # Seed for reproducibility in Ollama
        )
        messages = [
            ("system", system_message),
            ("human", user_message),
        ]
        response = ollama_llm.invoke(messages)
        content = response.content
    
    else:
        raise ValueError(f"Unsupported api_type: {api_type}")

    return content


def summarise_cpd_priority(text: str, country: str = "", **kwargs) -> str:
    system_message = f"""
    You are a UNICEF expert working on Country Programme Evaluations. Your task is provide a succinct and 
    accurace summary of country priorities based on the extract from {country} Country Programme Document (CPD).
    Respond with no more than 3 bullet points. Your response should be concise yet detailed, assuming that it 
    will be read by domain experts with sufficient background knowledge of the country.
    """
    summary = prompt(text, system_message, **kwargs)
    return summary


def summarise_coar_context(text: str, country: str = "", number_of_paragraphs: str = "two", paragraph_length: str = "100", **kwargs) -> str:
    system_message = f"""
    You are a UNICEF expert working on Country Programme Evaluations. Your task is to summarize sections from {country} Country Office Annual Reports to describe the context and situation of children in the country.

	- Synthesize the country context into {number_of_paragraphs} concise paragraphs (no more than {paragraph_length} words each) with a flowing, cohesive narrative. Do **not use** titles or headers.
	- Provide a chronological account of key events, highlighting significant milestones and examples from each available year.
	- Use **bold text** to emphasize **key events**, **dates**, and **topics** related to Health, Nutrition, COVID-19, Child Protection, WASH, Education, Gender Equality, Social Protection and Inclusion, Humanitarian Action, Early Childhood Development (ECD), Adolescent Development, Vulnerable Groups, and Children with Disabilities.
	- Ensure the summary covers information for as many years as you can find.
	- Prioritize critical information, avoiding unnecessary details, and ensure smooth transitions between topics and years.
	- Use clear, professional language that is accessible to a general audience.
    - Use Markdown syntax. If using headers, only use '####' and '#####'.
    """
    summary = prompt(text, system_message, **kwargs)
    return summary


def summarise_coar_contributions(text: str, country: str = "", **kwargs) -> str:
    system_message = f"""
    You are a UNICEF expert working on Country Programme Evaluations. Your task is to summarise sections from 
    {country} Country Office Annual Reports to describe major contributions and drivers of results of the country programme.
    
    - Do not provide a chronological summary. Instead, synthesise information from all years to identify key contributions 
    and drivers of results. Provide examples of main successes and challenges, if applicable.
    - Be sure to include years (if available) 2018, 2019, 2019, 2020, 2021, 2022 and 2023.
    - Your response must be at most two paragraphs.
    - Emphasise key events, figures and dates in **bold**. 
    - Use Markdown syntax. If using headers, only use '####' and '#####'.
    """
    summary = prompt(text, system_message, **kwargs)
    return summary



def summarise_innovations(text: str, country: str = "", **kwargs) -> str:
    system_message = f"""
    You are a UNICEF expert working on Country Programme Evaluations. Your task is to summarise sections from 
    {country} reports to describe innovative strategies and novel approaches adopted by the country office.

    Instructions:

        1.	Summarize chronologically with clear, concise information.
        2.	**Identify and list specific examples** of innovations as thoroughly as possible.
        3.	Organize these examples into **clear bullet points** (using '- ').
        4.	Within each bullet point, highlight key terms or concepts in **bold**.
        5.	Provide **concrete examples** of innovations or novel approaches for each point.
        6.	Focus solely on **innovative strategies and novel approaches**, avoiding unrelated content.
        7.	**Do not repeat** the same example of innovation if already mentioned.
        8.	Format the response using Markdown syntax for readability. **Do not use headers**.

    Response Format Example:

        - **Innovation 1**: Example description.
        - **Innovation 2**: Example description.
        - ...
        - **Innovation N**: Example description.
    """
    summary = prompt(text, system_message, **kwargs)
    return summary


def summarise_gender(text: str, country: str = "", **kwargs) -> str:
    system_message = f"""
You are a UNICEF expert conducting Country Programme Evaluations. Your task is to summarize sections from reports for {country} that describe approaches related to gender equality and/or the empowerment of girls and women.

Instructions:

	1. Summarize chronologically: Provide clear, concise information in the order events or activities occurred.
	2. Highlight specific examples: Thoroughly identify and list examples directly related to gender equality and/or the empowerment of girls and women.
	3. Organize examples into bullet points:
	    * Use ’- ’ for each bullet point.
	    * Highlight key terms or concepts using bold text.
	4. Provide concrete examples:
	    * Include specific activities, outcomes, or initiatives where applicable.
	5. Stay focused: Only include information relevant to gender equality and/or the empowerment of girls and women. Exclude unrelated content.
	6. Avoid repetition: Do not repeat the same example in different bullet points.
	7. Use Markdown for formatting: Ensure the response is visually clear and easy to read. Avoid headers or unnecessary formatting.

    Response Format Example:

        - **Gender Activity 1**: Example description.
        - **Gender Activity  2**: Example description.
        - ...
        - **Gender Activity  N**: Example description.
    """
    summary = prompt(text, system_message, **kwargs)
    return summary


def summarise_coar_innovations(text: str, country: str = "", **kwargs) -> str:
    system_message = f"""
    You are a UNICEF expert working on Country Programme Evaluations. Your task is to summarise sections from 
    {country} Country Office Annual Reports to describe innovative strategies and novel approaches adopted by the country office.

    - Your response should provide a succinct chronological summary with conrete examples of innovations.
    - You can list up to three bullet points per year.
    - Within each point, emphasise key things in **bold**.
    - Use Markdown syntax. If using headers, only use '####' and '#####'.

    

    ### Response Format

    **Year 1**

    - Point 1
    - Point 2

    **Year 2**

    - Point 1
    
    ...

    **Year n**

    - Point 1
    - Point 2
    - Point 3
    """
    summary = prompt(text, system_message, **kwargs)
    return summary


def summarise_coar_innovations_by_year(text: str, country: str = "", **kwargs) -> str:
    system_message = f"""
    You are a UNICEF expert working on Country Programme Evaluations. Your task is to summarise sections from 
    {country} Country Office Annual Reports to describe innovative strategies and approaches adopted by the country office.

	- Provide a succinct chronological summary of the key innovations, using concrete examples.
	- Limit your response to up to 10 bullet points.
	- Highlight key details within each point using bold text.
	- Format your response in Markdown.

Response Format:

    - Brief description with key details.
	- Brief description with key details.

    """
    summary = prompt(text, system_message, **kwargs)
    return summary

def summarise_coar_partnerships(text: str, country: str = "", **kwargs) -> str:
    system_message = f"""
     You are a UNICEF expert working on Country Programme Evaluations. Your task is to summarize sections from {country} Country Office Annual Reports and Regional Director Letters to describe the partnerships and collaborations of the country office.
        - Provide a succinct chronological summary of partnerships and collaborations, incorporating concrete examples.
        - Limit your response to no more than 2 short paragraphs.
        - Cite the year of each example in **bold**.
        - Emphasize key partners and projects in **bold**.
        - Ensure the narrative flows smoothly and is easy to read.
        - Maintain a professional tone suitable for a general audience.
        - Use Markdown syntax. If using headers, only use '####' and '#####'.
    """
    summary = prompt(text, system_message, **kwargs)
    return summary

def summarise_coar_partnerships_v2(text: str, country: str = "", **kwargs) -> str:
    system_message = f"""
    You are a UNICEF expert working on Country Programme Evaluations. Your task is to summarise sections from 
    {country} Country Office Annual Reports to describe partnerships and collaborations of the country office.

    - Write a cohesive narrative in **Markdown** format, consisting of **one short paragraph** for each year, no longer than **150 words**, without titles or headers.
    - Include information from **all available years**, providing a comprehensive overview by mentioning at least **one key event for every year** in **chronological order**.
    - Highlight the year at the beginning of each paragraph in **bold**.
    - Use clear and accessible language suitable for a general audience while maintaining a professional tone.
    - Focus on concisely presenting the most critical information, avoiding unnecessary details.
    - **Extract and emphasize** as many **partners and projects** as you find by making them **bold** within the text.
    """
    summary = prompt(text, system_message, **kwargs)
    return summary


def summarise_coar_innovation_v2(text: str, country: str = "", **kwargs) -> str:
    system_message = f"""
        You are a UNICEF expert conducting Country Programme Evaluations. Your task is to summarize the key innovations and innovative models from the {country} Country Office Annual Reports.

        - Compose a **cohesive narrative** in **Markdown** format, with a **single concise paragraph** for each year, keeping each paragraph under **150 words**.
        - Cover **all available years** in **chronological order**, ensuring that each year highlights at least **one significant event or initiative**.
        - Begin each paragraph with the year in **bold**.
        - Use clear, professional language accessible to a general audience, avoiding jargon or excessive details.
        - Focus on presenting the **most impactful innovations**, emphasizing **key partners and projects** by making them **bold** in the text.
        - Ensure the summary provides a **comprehensive overview** of the country office's innovations across the years.
    """
    summary = prompt(text, system_message, **kwargs)
    return summary


def summarise_sitans_recommendations(text: str, country: str = "", **kwargs) -> str:
    system_message = f"""
    You are a UNICEF expert working on Country Programme Evaluations. Your task is to summarise sections from
    {country} Situational Analysis (SitAns) focusing on key recommendations and take-away policy messages.

    - Your response should be a set of 3-5 short bullet points succinctly summarising the recommendations.
    - Emphasise key partners and projects in **bold**.
    - Use Markdown syntax. If using headers, only use '####' and '#####'.
    """
    summary = prompt(text, system_message, **kwargs)
    return summary



def extract_by_goal_area(text: str, subject: str = "needs and challenges", **kwargs) -> str:
    system_message = f"""
    You are a UNICEF expert specializing in Country Programme Evaluations. Your task is to analyze each block of text and identify the main **{subject}**. Categorize these into one or more of the following goal areas:

    - **Survive and Thrive**: Every child, including adolescents, survives and thrives, with access to adequate diets, services, practices and supplies: ECD, Health, HIV/AIDS, Nutrition.
    - **Learn**: Every child, including adolescents, learns and acquires skills for the future: Education.
    - **Protection from Violence and Exploitation**: Every child, including adolescents, is protected from violence, exploitation, abuse, neglect and harmful practices: Child Protection.
    - **Safe and Clean Environment**: Every child, including adolescents, has access to safe and equitable water, sanitation and hygiene services and supplies, and lives in a safe and sustainable climate and environment: WASH.
    - **Equitable Chance in Life**: Every child, including adolescents, has access to inclusive social protection and lives free of poverty: Social Protection.
    - **Cross Sectoral**: Improved and coordinated action for achieving multiple sectoral outcomes. This includes integrated planning, programme reviews, and robust monitoring and data analysis. The focus also extends to social and behaviour change, logistics and supply chains, and thorough evaluation and research activities. Cross-sectoral efforts address critical issues like gender-discriminatory roles and practices, support for children with disabilities, conflict prevention, peacebuilding, and adherence to human rights standards. It also includes communication and advocacy initiatives, along with operational support to enhance program delivery.    
    - **Development Effectiveness**: Higher quality programmes through results-based management and improved accountability of results
    - **Management**: Enhanced management of financial, human, and operational resources to achieve results. This includes corporate financial oversight, robust information and communication technology (ICT) systems, and administrative management. It covers external relations, partnerships, communication, and resource mobilization to support UNICEF's goals. Leadership and strategic corporate direction are key, along with ensuring the safety and security of staff and premises. Effective management also extends to oversight, support, and operations for field and country offices, ensuring alignment with organizational standards and goals.
    - **UN Coordination**: UN development system leadership and coordination in assigned humanitarian clusters.
    - **Special Purpose**: Special purpose including capital investments, private-sector fundraising, procurement services, and others.

    ### Important:
    - The text following each goal area (in **bold**) is meant to help you understand its scope. **Do not use it directly for categorization.** Base your categorization solely on the block of text being analyzed.

    ### For each block of text:
    1. Identify **as many specific {subject} as possible**.
    2. Organize the identified {subject} into **clear bullet points** ('- ').
    3. Begin each bullet point with the goal area in **bold**, followed by a concise description of the {subject}.
    4. Include **only relevant goal areas** based on the content.
    5. **Repeat goal areas** if multiple {subject} are identified under the same category.
    6. Avoid unrelated information, focusing only on **relevant {subject}**.
    7. **Avoid repeating** the same {subject} if they are already mentioned.
    8. Format the output using **Markdown syntax**.

    ### Example Output:
    - **Survive and Thrive**: High neonatal mortality rates require immediate attention.
    - **Learn**: Low preschool enrollment in underserved areas, particularly rural regions.
    - **Protection from Violence and Exploitation**: Increased rates of child abuse during protests.
    """
    summary = prompt(text, system_message, **kwargs)
    return summary


def classify_coar_sections(text: str, **kwargs) -> str:
    system_message = (
        'You are a UNICEF expert working on Country Programme Evaluations. '
        'Your task is to classify the following excerpts from Country Office Annual Reports into one of the following categories. '    
        'Reply only with the relevant category name, nothing more, name only, no markdown. '
        'If it is impossible to assign a category, reply with "No category.".\n'
        '\n'
        'The categories are:\n'
        '**Update on the context and situation of children**\n'
        '**Major contributions and drivers of results**\n'
        '**UN Collaboration and Other Partnerships**\n'
        '**Lessons Learned and Innovations**\n'
    )

    category = prompt(text, system_message, **kwargs)
    return category


def classify_text(text: str, system_message: str, **kwargs) -> int:
    """
    Check if the text is relevant to the evaluation criteria and question.
    Args:
        text (str): The text to classify.
        prompt (str): The classification prompt. Shouldd return an integer.
    Returns:
        int: whatever is ddefined, -1 when unsure.
    """
    
    response = prompt(text, system_message, **kwargs)
    try:
        response = int(response)
    except ValueError:
        response = -1
    return response


def summarize_factors_affecting_delivery_results(text: str, country: str, **kwargs) -> str:
    system_message = "You are a UNICEF expert working on Country Programme Evaluations. " \
        f"Your task is to summarize sections from different documents for {country} focusing on internal and external factors affecting the delivery of results for children by UNICEF.\n" \
        "\n"\
        "- Provide a succinct summary on factors affecting the delivery of results for children by UNICEF.\n"\
        "- Differentiate between external and internal factors.\n"\
        "- Limit your response to no more than 2 paragraphs.\n"\
        "- Cite the year of each example in **bold**.\n"\
        "- Emphasize key factors in **bold**.\n"\
        "- Use Markdown syntax. If using headers, only use '####' and '#####'.\n"\
        "- Ensure the narrative flows smoothly and is easy to read.\n"\
        "- Maintain a professional tone suitable for a general audience.\n"\
        
    response = prompt(text, system_message, **kwargs)

    return response
    
def summarize_types_of_stakeholders(text: str, country: str, **kwargs) -> str:
    system_message = "You are a UNICEF expert working on Country Programme Evaluations. " \
        f"Your task is to summarize sections from different documents for {country} focusing on types of stakeholders connected with UNICEF through formal and informal links.\n" \
        "\n"\
        "- Provide a succinct summary on types of stakeholders connected with UNICEF through formal and informal links.\n"\
        "- Differentiate between different types of stakeholders.\n"\
        "- Limit your response to no more than 2 paragraphs.\n"\
        "- Cite the year of each example in **bold**.\n"\
        "- Emphasize key stakeholders in **bold**.\n"\
        "- Use Markdown syntax. If using headers, only use '####' and '#####'.\n"\
        "- Ensure the narrative flows smoothly and is easy to read.\n"\
        "- Maintain a professional tone suitable for a general audience.\n"\
        
    response = prompt(text, system_message, **kwargs)

    return response
    
def summarize_types_of_advocacy_for_stakeholders(text: str, country: str, **kwargs) -> str:
    system_message = "You are a UNICEF expert working on Country Programme Evaluations. " \
        f"Your task is to summarize sections from different documents for {country} focusing on types of advocacy, programmatic or other work for the different stakeholders connected with UNICEF.\n" \
        "\n"\
        "- Provide a succinct summary on types of advocacy, programmatic or other work performed by UNICEF on it'sa stakeholders (formal or informal).\n"\
        "- Limit your response to no more than 2 paragraphs.\n"\
        "- Cite the year of each example in **bold**.\n"\
        "- Emphasize key stakeholders in **bold**.\n"\
        "- Use Markdown syntax. If using headers, only use '####' and '#####'.\n"\
        "- Ensure the narrative flows smoothly and is easy to read.\n"\
        "- Maintain a professional tone suitable for a general audience.\n"\
        
    response = prompt(text, system_message, **kwargs)

    return response
    
def summarize_policy_changes(text: str, country: str, **kwargs) -> str:
    system_message = "You are a UNICEF expert working on Country Programme Evaluations. " \
        f"Your task is to summarize sections from different documents for {country} focusing on types policy changes affected by UNICEF." \
        "\n"\
        "- Provide a succinct summary on types of policy changes affected by UNICEF.\n"\
        "- Limit your response to no more than 2 paragraphs.\n"\
        "- Cite the year of each example in **bold**.\n"\
        "- Emphasize key policy changes in **bold**.\n"\
        "- Use Markdown syntax. If using headers, only use '####' and '#####'.\n"\
        "- Ensure the narrative flows smoothly and is easy to read.\n"\
        "- Maintain a professional tone suitable for a general audience.\n"\
        
        
    response = prompt(text, system_message, **kwargs)

    return response
    
def summarize_unicef_positioning(text: str, country: str, **kwargs) -> str:
    system_message = "You are a UNICEF expert working on Country Programme Evaluations. " \
        f"Your task is to summarize sections from different documents for {country} focusing on how UNICEFs positioning in the general public, CSOs , and private sector enables the delivery of results for children.\n" \
        "\n"\
        "- Provide a succinct summary on on how UNICEFs positioning in the general public, CSOs , and private sector enables the delivery of results for children.\n"\
        "- Differentiate between different types of stakeholders.\n"\
        "- Limit your response to no more than 2 paragraphs.\n"\
        "- Cite the year of each example in **bold**.\n"\
        "- Emphasize key stakeholders in **bold**.\n"\
        "- Use Markdown syntax. If using headers, only use '####' and '#####'.\n"\
        "- Ensure the narrative flows smoothly and is easy to read.\n"\
        "- Maintain a professional tone suitable for a general audience.\n"\
        
    response = prompt(text, system_message, **kwargs)

    return response


    
def summarize_types_of_advocacy_for_cso_stakeholders(text: str, country: str, **kwargs) -> str:
    system_message = "You are a UNICEF expert working on Country Programme Evaluations. " \
        f"Your task is to summarize sections from different documents for {country} focusing on types of advocacy, programmatic or other work done by UNICEF for the public, Civil Society Organizations and other private sector stakeholders connected with UNICEF through formal or informal links.\n" \
        "\n"\
        "- Provide a succinct summary on types of advocacy, programmatic or other work done by UNICEF for the public, Civil Society Organizations and other private sector stakeholders connected with UNICEF through formal or informal links.\n"\
        "- Limit your response to no more than 2 paragraphs.\n"\
        "- Cite the year of each example in **bold**.\n"\
        "- Emphasize key stakeholders in **bold**.\n"\
        "- Use Markdown syntax. If using headers, only use '####' and '#####'.\n"\
        "- Ensure the narrative flows smoothly and is easy to read.\n"\
        "- Maintain a professional tone suitable for a general audience.\n"\
        
    response = prompt(text, system_message, **kwargs)

    return response


def classify_goal_area_into_result_area(text: str, goal_area: str, **kwargs) -> str:
    # Define the goal areas and their corresponding result areas
    goal_areas_to_result_areas = {
        "Survive and Thrive": [
            'Strengthening primary health care and high-impact health interventions',
            'Immunization services as part of primary health care',
            'Fast-track the end of HIV/AIDS',
            'Health and development in early childhood and adolescence',
            'Mental health and psychosocial well-being',
            'Nutrition in early childhood',
            'Nutrition of adolescents and women',
            'Early detection and treatment of malnutrition',
        ],
        "Learn": [
            'Access to quality learning opportunities',
            'Learning, skills, participation and engagement'
        ],
        "Protection from Violence and Exploitation": [
            'Protection from violence, exploitation, abuse and neglect',
            'Promotion of care, mental health and psychosocial well-being and justice',
            'Prevention of harmful practice',
        ],
        "Safe and Clean Environment": [
            'Safe and equitable water, sanitation and hygiene services and practices',
            'Water, sanitation and hygiene systems and empowerment of communities',
            'Climate change, disaster risks and environmental degradation',
        ],
        "Equitable Chance in Life": [
            'Reducing child poverty',
            'Access to inclusive social protection',
        ],
        "Cross Sectoral": [
            'Cross-sectoral - planning and programme reviews',
            'Cross-sectoral - Monitoring, data and situation analyses',
            'Cross-sectoral - Social and behaviour change',
            'Supply and logistics',
            'Evaluation',
            'Research',
            'Cross-sectoral - Other programme areas',
            'Cross sectoral - Gender discriminatory roles and practices',
            'Cross sectoral - Children with Disabilities',
            'Cross-sectoral - Conflict prevention, fragility and peacebuilding',
            'Cross-sectoral - Human rights',
            'Communication and Advocacy',
            'Operations support to programme delivery',
        ],
        "Development Effectiveness": [
            'Technical excellence in policy and programmes',
            'Technical excellence in procurement and management of supplies',
            'Technical excellence in humanitarian action',
        ],
        "Management": [
            'Corporate financial, information and communication technology and administrative management',
            'Corporate external relations and partnerships, communication and resource mobilization',
            'Leadership and corporate direction',
            'Staff and premises security',
            'Field/country office oversight, management and operations support',
        ],
        "UN Coordination": ['United Nations coherence and cluster coordination'],
        "Special Purpose": [
            'PSFR modalities non-post',
            'PSFR Technical assistance',
            'Private sector engagement',
            'Procurement services',
            'Capital investments',
        ]
    }

    # Retrieve the list of result areas for the specified goal area
    result_areas = goal_areas_to_result_areas.get(goal_area, [])
    
    if not result_areas:
        raise ValueError(f"Invalid goal area provided: {goal_area}")

    # Format result areas in bold and bullet-point separated

    formatted_result_areas = "\n"+"\n".join([f"     - **{area}**" for area in result_areas])

    # Define the system message for the prompt with formatted result areas
    system_message = f"""
    You are a UNICEF expert. Analyze the following text and classify it into **only one** result area based on the provided list of result areas:
    {formatted_result_areas}

    Your task is to read the text and return **exactly one** result area from the list that best matches the content.

    ### Important Instructions:
    - You must carefully select the **most appropriate result area** from the predefined list based on the content of the text.
    - **Do not invent or combine result areas**. The result area should match **exactly** one of the predefined options.
    - **Do not provide explanations or return multiple result areas**—the output should be **only the name** of the selected result area.
    - Return **only** the chosen result area.

    ### Example:

    **Text**: "Decreasing budget allocations for health, with health spending among the lowest in the region."
    **Expected Output**: "Strengthening primary health care and high-impact health interventions"

    **Text**: "High prevalence of overweight among children under 5 years and low exclusive breastfeeding rates."
    **Expected Output**: "Nutrition in early childhood"
    """
    # Use the provided `prompt` function to generate the response
    classification = prompt(user_message=text, system_message=system_message, **kwargs)

    return classification.strip()


def summarise_acomplishments(text: str, **kwargs) -> str:
    system_message = f"""
    You are a UNICEF expert working on Country Programme Evaluations. Your task is to summarize sections from reports to identify key strengths and areas for improvement. Your goal is to categorize them into two groups: **"Strengths"** (well-executed areas) and **"Areas for Improvement"** (sections needing further attention or action).

    ### Instructions:

    1. **Identify and list** as many specific strengths and areas for improvement as possible, focusing only on relevant information.
    2. Organize the identified points into two categories: **Strengths** (well-executed areas) and **Areas for Improvement** (needs further action).
    3. Use **clear bullet points** (`-`) under each category and **start each bullet** with the **bolded section name** it refers to, followed by a concise description of the strength or area for improvement.
    4. **Repeat the section name** if multiple strengths or areas for improvement are identified under the same category.
    5. **Do not repeat** the same strengths or areas for improvement within the list.
    6. Focus strictly on the **key strengths and areas for improvement** relevant to the evaluation, avoiding unrelated details.
    7. Format the output using **Markdown syntax**.

    ### Example Output:

    #### Strengths**
    - **Partnerships1**: Example description.
    - **Partnerships2**: Example description.
    - **Management and Operations**: Example description.
    - …

    #### Areas for Improvement**
    - **Partnerships1**: Example description.
    - **Management and Operations**: Example description.
    - **ProgrammingN**: Example description.
    """
    summary = prompt(text, system_message, **kwargs)
    return summary


def summarise_comparative_advantage(text: str, **kwargs) -> str:
    system_message = f"""
    You are a UNICEF expert working on Country Programme Evaluations. 
    Your task is to summarize sections from evaluation reports to identify key UNICEF’s comparative advantage. 

    ### Instructions:

    1. **Identify and list** as many comparative advantages as possible, focusing only on relevant information.
    2. Use **clear bullet points** (`-`) under each category and **start each bullet** with the **bolded section name** it refers to, followed by a concise description of the comparative advantage.
    4. **Repeat the section name** if multiple comparative advantages are identified under the same title.
    5. **Do not repeat** the same comparative advantage within the list.
    6. Focus strictly on the **UNICEF’s comparative advantage** relevant to the evaluation, avoiding unrelated details.
    7. Format the output using **Markdown syntax**.

    ### Example Output:

    - **Comparative advantage1**: Example description.
    - **Comparative advantage2**: Example description.
    - **Comparative advantage3**: Example description.
    - …

    """
    summary = prompt(text, system_message, **kwargs)
    return summary