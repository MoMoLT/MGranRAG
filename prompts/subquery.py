query1 = """As a knowledge graph expert, I need to extract relevant entities and their relationships from a question to facilitate information retrieval in a knowledge graph.

For example, consider the question: 
Which state does the drug stores, of which the CEO is Warren Bryant, are located?

From the question, we can identify four key entities with their types:
- durg stores[&organization&]
- Warren Bryant[&person&]
- CEO[&role&]
- state[&location&]

Relationships form a connected chain:
- chain 1: Warren Bryant → CEO → durg stores → state

Break down the relationships into simple triplets:
relation1:
    - relation_type: person_position
    - subquery: Who is the CEO of the drug stores?
    - template: [Warren Bryant] holds the position of [CEO].
relation2:
    - relation_type: position_organization
    - subquery: What drug stores have a CEO position?
    - template: The [CEO] is a role at [drug stores].
relation3:
    - relation_type: organization_location
    - subquery: Where are the drug stores located?
    - template: The [drug stores] are located in [state].

# Output Format:
```
# Extract Entities
- entityX[&type&]
...

# Relationships chain
- chain 1: entityX → entityY → entityZ
- ...

# Generate Relationship
relationX:
    - relation_type: relationship type
    - subquery: Subsidiary query to refine the main query
    - template: Natural language template with [entities]
...
```

Next, please address this question:
Which university does the AI lab, of which the director is Sophia Patel, belongs to?
"""
reply1 = """# Extract Entities
- AI lab[&organization&]
- Sophia Patel[&person&]
- director[&role&]
- university[&organization&]

# Relationships chain
- chain 1: Sophia Patel → director → AI lab → university

# Generate Relationship
relation1:
    - relation_type: person_position
    - subquery: Who is the director of the AI lab?
relation2:
    - relation_type: position_organization
    - subquery: What AI lab has a director position?
relation3:
    - relation_type: organization_institution
    - subquery: Which university does the AI lab belong to?
"""
query2 = """Comparative questions require extracting comparative information for each entity. For example:
Which magazine was started first, Arthur's Magazine or First for Women?

# Step 1: Extract Entities
- Arthur's Magazine [&publication&]
- First for Women [&publication&]
- publication date [&temporal&]

# Step 2: Relationships chain
- chain 1: Arthur's Magazine → publication date
- chain 2: First for Women → publication date

# Step 3: Generate Relationship
relation1:
    - type: publication_time
    - subquery: When was Arthur's Magazine first published?
relation2:
    - type: publication_time
    - subquery: When was First for Women first published?

Next, please address this question:
Who is older, Leonardo DiCaprio or Brad Pitt?
"""
reply2 = """# Extract Entities
- Leonardo DiCaprio [&person&]
- Brad Pitt [&person&]
- birth date [&temporal&]

# Relationships chain
- chain 1: Leonardo DiCaprio → birth date
- chain 2: Brad Pitt → birth date

# Generate Relationship
relation1:
    - type: birth_date
    - subquery: When was Leonardo DiCaprio born?
relation2:
    - type: birth_date
    - subquery: When was Brad Pitt born?
"""
query3 = """Next, please address this question:
Who created an amusement park in Midnight City?
"""
reply3 = """# Extract Entities
- amusement park [&organization&]
- Midnight City [&location&]
- creator [&person&]

# Relationships chain
- chain 1: amusement park → creator
- chain 2: amusement park → Midnight City

# Generate Relationship
relation1:
    - type: creator_organization
    - subquery: Who created the amusement park?
    - template: [creator] is the creator of [amusement park].
relation2:
    - type: organization_location
    - subquery: Where is the amusement park located?
    - template: The [amusement park] is located in [Midnight City].
"""
query4 = """Please address this question:
Who is the current president of France?
"""
reply4 = """# Extract Entities
- France [&location&]
- president [&role&]
- [person] [&person&]

# Relationships chain
- chain 1: France → president → [person]

# Generate Relationship
relation1:
    - relation_type: position_organization
    - subquery: What country has the president role?
relation2:
    - relation_type: person_position
    - subquery: Who is the current president of France?
"""
query5 = """Which planet is larger, Mars or Venus?"""
reply5 = """# Extract Entities
- Mars[&celestial_body&]  
- Venus[&celestial_body&]  
- diameter[&measurement&]

# Relationships chain
- chain 1: Mars → diameter
- chain 2: Venus → diameter

# Generate Relationship
relation1:
    - relation_type: celestial_body_measurement
    - subquery: What is the diameter of Mars?
    - template: [Mars] has a diameter of [diameter].
relation2:
    - relation_type: celestial_body_measurement
    - subquery: What is the diameter of Venus?
    - template: [Venus] has a diameter of [diameter].

# Combined Template
COMBINED_TEMPLATE{[Mars] has a diameter of [diameter], while [Venus] has a diameter of [diameter].}

# Rewritten Query
REWRITTEN_QUERY{Which planet has the greater size between Mars and Venus?}
"""
query7 = """Why does global warming intensify the strength of hurricanes?"""
question_input = """Please address this question:
${question}

/no_think
"""
prompt_template = [
    {"role": "user", "content": query1},
    {"role": "assistant", "content": reply1},
    {"role": "user", "content": query2},
    {"role": "assistant", "content": reply2},
    # {"role": "user", "content": query3},
    # {"role": "assistant", "content": reply3},
    {"role": "user", "content": query4},
    {"role": "assistant", "content": reply4},
    # {"role": "user", "content": query5},
    # {"role": "assistant", "content": reply5},
    {"role": "user", "content": question_input}
]
