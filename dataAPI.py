import pandas as pd
import numpy as np
import time
from neuprint import Client, fetch_traced_adjacencies
#Pyarrow needs to be imported too

def fetch_neuron_data(auth):
    """
    Fetch neuron data from a specified neuprint server.

    Returns
    -------
    traced_df : DataFrame
        The DataFrame that contains traced neuron adjacencies and additional neuron data.
    total_conn_df : DataFrame
        The DataFrame that contains the total connection weights between pairs of neurons.

    Notes
    -----
    - The neuprint server requires a valid token for access.
    - Fetching data from the server can take some time depending on the network connection and the server response.

    Examples
    --------
    traced_df, total_conn_df = fetch_neuron_data()
    """

    client = Client('neuprint.janelia.org', 'manc:v1.0',
                    token=auth)
    traced_df, roi_conn_df = fetch_traced_adjacencies('manc-traced-adjacencies-v1.0')

    unique_body_ids = traced_df["bodyId"].unique().tolist()
    body_id_list_str = ', '.join([str(id) for id in unique_body_ids])

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
    start_time_fetch = time.time()
    neuron_df = client.fetch_custom(q)
    end_time_fetch = time.time()
    print(f"Time taken by client.fetch_custom: {end_time_fetch - start_time_fetch} seconds")

    # Merge traced_df with neuron_df
    traced_df = pd.merge(traced_df, neuron_df, on='bodyId')

    # Sum up over ROIs
    total_conn_df = roi_conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()

    return traced_df, total_conn_df

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

    # Create a new DataFrame with an empty column name for 'TempValues' and 'Completed' column filled with 'True'
    export_df = pd.DataFrame()
    export_df[''] = tempDF['TempValues']
    export_df['Completed'] = True

    # Save the DataFrame to a csv file
    export_df.to_csv('MANC_Data\\2023_06_06_completeness_1.0_final.csv', index=False)

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
    traced_df, total_conn_df = fetch_neuron_data(auth)
    process_neuron_data(traced_df, total_conn_df)

