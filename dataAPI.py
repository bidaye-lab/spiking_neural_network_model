import pandas as pd
import numpy as np
import time
from neuprint import Client, fetch_neurons, NeuronCriteria, queries
#Pyarrow needs to be imported too

def fetch_completeness_df(auth):
    """
    Fetches information about neurons from a NeuPrint server, filtering neurons based on their status, 
    and creates a DataFrame indicating the 'completeness' of neurons.

    Parameters:
    auth (str): The authentication token for the NeuPrint server.

    Returns:
    pandas.DataFrame: A DataFrame with bodyId and a column indicating the completeness of the neuron.
    """

    # Set up the client with the provided auth token
    client = Client('neuprint.janelia.org', 'manc:v1.0', token=auth)

    # Define criteria for neurons - we only want those with status 'Traced' or 'Unimportant'
    criteria = NeuronCriteria(status=['Traced', 'Unimportant'])

    # Fetch the neurons that match the criteria from the NeuPrint server
    singleNeurons, throwaway = fetch_neurons(criteria, client=client)

    # Create a new DataFrame with the bodyIds of the fetched neurons and a 'completed' column, 
    # which is filled with 'True' (indicating that these neurons are completed)
    completeness_DF = pd.DataFrame({
        '': singleNeurons['bodyId'],
        'completed': [True] * len(singleNeurons)
    })

    # Write the DataFrame to a CSV file
    completeness_DF.to_csv('MANC_Data\\2023_06_06_completeness_1.0_final.csv', index=False)

    return completeness_DF

def fetch_connectivity_df(auth):
    """
    Fetches connectivity data from a NeuPrint server, filtering neurons based on their status.

    Parameters:
    auth (str): The authentication token for the NeuPrint server.

    Returns:
    pandas.DataFrame: A DataFrame with the summed synaptic weights for each pair of neurons.
    """

    # Set up the client with the provided auth token
    client = Client('neuprint.janelia.org', 'manc:v1.0', token=auth)

    # Define criteria for source and target neurons - we only want those with status 'Traced' or 'Unimportant'
    source_criteria = NeuronCriteria(status=['Traced', 'Unimportant'])
    target_criteria = NeuronCriteria(status=['Traced', 'Unimportant'])

    # Use fetch_adjacencies with the defined source and target criteria to fetch connectivity data
    _, conn_df = queries.fetch_adjacencies(sources=source_criteria, targets=target_criteria, client=client)
    
    # Group the data by presynaptic and postsynaptic body IDs and sum the synaptic weights
    # This gives us a total synaptic weight for each pair of neurons
    summed_ROI_conn_df = conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()

    return summed_ROI_conn_df

def addPredictedNT(auth, completeness_DF):
    """
    Appends predicted neurotransmitter (NT) data to an existing DataFrame.
    
    Parameters
    ----------
    auth : str
        Authentication token for the NeuPrint client.
    completeness_DF : pd.DataFrame
        DataFrame containing initial neuron data. Assumes it contains a column with bodyId.
        
    Returns
    -------
    newCompleteness_DF : pd.DataFrame
        The updated DataFrame containing original data plus added predicted neurotransmitter data.
    """
    # Establish a client connection to the NeuPrint database
    client = Client('neuprint.janelia.org', 'manc:v1.0', token=auth)
    
    # Extract unique neuron body IDs from the existing DataFrame
    unique_body_ids = completeness_DF[''].unique().tolist()

    # Prepare the list of unique body IDs for the query string
    body_id_list_str = ', '.join([str(id) for id in unique_body_ids])

    # Construct a Cypher query to fetch predicted neurotransmitter data for each unique neuron
    q = f"""\
            MATCH (n :Neuron)
            WHERE n.bodyId IN [{body_id_list_str}]
            RETURN n.bodyId AS bodyId,
            n.predictedNt AS predicted_neurotransmitter, 
            n.predictedNtProb AS predictedNTProb, 
            n.ntUnknownProb AS ntUnknownProb, 
            n.ntAcetylcholineProb AS AcetylcholineProb,
            n.ntGabaProb AS GabaProb,
            n.ntGlutamateProb AS GlutamateProb
        """
    
    # Track the time taken to execute the custom query
    start_time_fetch = time.time()
    neuron_df = client.fetch_custom(q)
    end_time_fetch = time.time()
    print(f"Time taken by client.fetch_custom: {end_time_fetch - start_time_fetch} seconds")

    # Rename the empty column header to 'bodyId'
    completeness_DF.rename(columns={'': 'bodyId'}, inplace=True)

    # Merge the existing DataFrame with the newly fetched neurotransmitter data
    newCompleteness_DF = pd.merge(completeness_DF, neuron_df, on='bodyId')

    return newCompleteness_DF

def process_neuron_data(traced_df, total_conn_df):
    """
    Process fetched neuron data and save results in parquet and csv formats.

    Parameters
    ----------
    traced_df : DataFrame
        The DataFrame that contains traced neuron adjacencies and additional neuron data.
    total_conn_df : DataFrame
        The DataFrame that contains the total connection weights between pairs of neurons.

    Returns
    -------
    None
    """
    
    new_df = total_conn_df.copy()

    # Merge new_df with traced_df on bodyId_pre
    new_df = pd.merge(new_df, traced_df, left_on='bodyId_pre', right_on='bodyId')

    # Create new sum of GabaProb and GlutamateProb
    new_df['GabaGlutamateProb'] = new_df['GabaProb'] + new_df['GlutamateProb']

    # Create condition for GabaGlutamateProb > 0.5
    cond1 = new_df['GabaGlutamateProb'] > 0.5

    # Create condition for max of AcetylcholineProb and ntUnknownProb
    cond2 = new_df[['AcetylcholineProb', 'ntUnknownProb']].idxmax(axis=1) == 'AcetylcholineProb'

    # Create condition for no values in the four probability columns
    cond3 = new_df[['ntUnknownProb', 'AcetylcholineProb', 'GabaProb', 'GlutamateProb']].isna().all(axis=1)

    # Calculate weight_modified based on conditions
    new_df['weight_modified'] = np.where(cond1, new_df['weight'] * -1,
                                         np.where(cond2, new_df['weight'] * 1,
                                                  np.where(cond3, new_df['weight'], new_df['weight'])))

    # Set status based on conditions
    new_df['status'] = np.where(cond1 | cond2, 'SOLVED', 'UNSOLVED')

    # Renaming columns and create new ones
    new_df.rename(columns={'bodyId_pre': 'Presynaptic_ID', 'bodyId_post': 'Postsynaptic_ID', 'weight': 'Connectivity', 'weight_modified': 'Excitatory x Connectivity'}, inplace=True)
    new_df['Unnamed'] = new_df.index

    tempDF = pd.DataFrame()
    tempDF['TempValues'] = pd.concat([new_df['Presynaptic_ID'], new_df['Postsynaptic_ID']], ignore_index=True)
    tempDF.drop_duplicates(subset='TempValues', inplace=True)
    labels, uniques = pd.factorize(tempDF['TempValues'])
    mapping_dict = dict(zip(uniques, labels))

    new_df['Presynaptic_Index'] = new_df['Presynaptic_ID'].map(mapping_dict)
    new_df['Postsynaptic_Index'] = new_df['Postsynaptic_ID'].map(mapping_dict)

    new_df['Excitatory'] = np.where(cond1, -1, np.where(cond2, 1, np.nan))

    # Rearrange columns according to the requirement
    new_df = new_df[['Unnamed', 'Presynaptic_ID', 'Postsynaptic_ID', 'Presynaptic_Index', 'Postsynaptic_Index', 'Connectivity', 'Excitatory', 'Excitatory x Connectivity', 'status']]
    
    #Re-define Presynaptic_ID and Postsynaptic_ID from DF Object Data type to int64
    new_df['Presynaptic_ID'] = new_df['Presynaptic_ID'].astype('int64')
    new_df['Postsynaptic_ID'] = new_df['Postsynaptic_ID'].astype('int64')

    # Save the dataframe to parquet and csv files
    new_df.to_parquet('MANC_Data\\2023_06_06_connectivity_1.0_final.parquet', compression='brotli', index = True)
    
    # Create a DataFrame with unique Presynaptic_IDs and their status
    status_df = new_df[['Presynaptic_ID', 'status']].drop_duplicates()

    # Convert 'status' from 'SOLVED'/'UNSOLVED' to True/False
    status_df['status'] = status_df['status'].map({'SOLVED': True, 'UNSOLVED': True})

    # Rename 'status' column to 'completed'
    status_df.rename(columns={'status': 'Completed'}, inplace=True)

def runAPIcall(auth):
    #Generate Completeness Data Frame and CSV file
    completeness_DF = fetch_completeness_df(auth)

    # Add Predicted Neurotransmitters to the Previous Completeness DF
    newCompleteness_DF = addPredictedNT(auth, completeness_DF)

    #Generate Connectivity Dataframe, Sorted by Region of Interest (RoI)
    summed_ROI_conn_df = fetch_connectivity_df(auth)

    # Process data to get Parquet File
    process_neuron_data(newCompleteness_DF, summed_ROI_conn_df)
