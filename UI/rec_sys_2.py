import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

def get_best_match_rapid(query, choices, scorer=fuzz.WRatio):
    """
    Returns the best match of `query` in `choices` using Rapidfuzz.
    - `scorer`: Scoring function from fuzz, such as fuzz.WRatio, fuzz.QRatio, etc.
    """
    result = process.extractOne(query, choices, scorer=scorer)
    if result:
        return result[0]
    return None


def ingredients_to_binary(ingredient_string, df_ingredients):
    """
    Converts a string of ingredients into a binary list indicating the presence of unique ingredients based on the best match.
    - `ingredient_string`: A string of ingredients separated by commas or spaces.
    - `unique_ingredients`: A list of all unique ingredient names.
    """
    # Initialize the binary list
    ingredients_binary = [0] * len(df_ingredients)
    
    # Normalize and split the ingredient string by commas or spaces
    queries = ingredient_string.replace(',', ' ').split()
    
    for query in queries:
        query = query.lower()
        best_match = get_best_match_rapid(query,df_ingredients)
        if best_match:
            # Update the binary list
            index = df_ingredients.index(best_match)
            ingredients_binary[index] = 1  # Set to 1 at the index of the best match

    return ingredients_binary


def top_n_scores(scores, n):
    # Flatten the array (if it's multidimensional)
    scores = scores.flatten()
    
    # Get the indices that would sort the array, and select the last 'n' indices (top scores)
    top_indices = np.argsort(scores)[-n:][::-1]
    
    
    return top_indices

def get_columns_with_ones(df, row_index):
    condition = df.loc[row_index] == 1
    return df.columns[condition].tolist()

def get_recipes(df, similarity, n):
    top_indices = top_n_scores(similarity, n)
    recipes = []
    ingredients = []

    for each in top_indices:
        recipe_name = df.loc[df['id'] == each, 'recipename'].values[0]
        recipes.append(recipe_name)
        ingredients.append(get_columns_with_ones(df, each))
        
    return recipes, ingredients

def rank_recipes_by_cuisine(recipes, ingredients, clusters, selected_cuisine):
    # Creating a DataFrame from recipes and ingredients
    recipes_df = pd.DataFrame({'Recipe': recipes, 'Ingredients': ingredients})
    # Set recipes as the index for easy filtering
    relevant_data = clusters.set_index('recipename').loc[recipes, [selected_cuisine]]
    # Merge on recipe names to align cuisine data
    recipes_df = recipes_df.set_index('Recipe')
    ranked_recipes_df = recipes_df.join(relevant_data).sort_values(by=selected_cuisine, ascending=False).reset_index()
    return ranked_recipes_df
    
def main():
    # Load data
    data = pd.read_csv('../recipe_ingredients_dataset/ingredient_df_with_recipenames.csv')

    # Get list of unique ingredients and cuisine styles from the dataset
    unique_ingredients = [col for col in data.columns if col not in ['id', 'cuisine', 'recipename']]
    cuisine_styles = data['cuisine'].unique().tolist()  # Assuming 'cuisine' is a column with categorical data

    # Load in SVD Matrices 
    vmatrix = pd.read_csv('../recipe_ingredients_dataset/V_1000matrix.csv',index_col=0)
    umatrix = pd.read_csv('../recipe_ingredients_dataset/U_1000matrix.csv',index_col=0)

    cluster = pd.read_csv('../recipe_ingredients_dataset/pca_clusters.csv',index_col=0)
    
    st.title("Ingredient Input Interface")

    
    # Text input for ingredients
    ingredients = st.text_input("Enter ingredients, separated by commas:")
    

    # Dropdown menu for cuisine styles 
    selected_cuisines = st.multiselect("Select cuisine style (max 1):", 
                                       cuisine_styles,
                                       default=None,
                                       help="You can select only one cuisine styles.")

    if len(selected_cuisines) > 1:
        st.error("Please select only one cuisine styles.")
        return 
  
    if st.button("Submit"):
                st.success("Submitted successfully!")
                # Process the user's ingredient input through the function
                x = ingredients_to_binary(ingredients, unique_ingredients)
                # Make sample a Series for manipulation
                x = pd.Series(x)
                # Dot product of user input and the V matrix. 
                xcon = x.dot(vmatrix)
                # xconcept and U must both be numpy arrays, and must be properly reshaped. 
                xcon = np.array(xcon)
                umatrix = np.array(umatrix)
                xcon = xcon.reshape(1, -1)
                #cosine similarity with our U matrix
                similarities = cosine_similarity(umatrix, xcon)
                # Get recommended recipes and ingredients 
                recipe_names, recipe_ingredients_lists = get_recipes(data, similarities, 30)
                
                if selected_cuisines:
                    
                    ranked_recipes_df = rank_recipes_by_cuisine(recipe_names, recipe_ingredients_lists, cluster, selected_cuisines[0])
                    st.write("Ranked Recipes based on Cuisine Preference:")
                    for i, row in enumerate(ranked_recipes_df.itertuples(), 1):
                        if i > 5:
                            break  # Break the loop after displaying the top 5 recipes
                        st.write(f"{i}. Recipe: {row.Recipe}")  # Corrected to use 'Recipe' column
                        st.write(f"   Ingredients: {', '.join(row.Ingredients)}\n")

                    
if __name__ == "__main__":
    main()


    
