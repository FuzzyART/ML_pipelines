import numpy as np
import pandas as pd
import json

def processGngModel(model_string,y_train):
    data = json.loads(model_string)
    points = None
    edges = None
    edge_positions = None
    rows = []
    for neuron in data["model"]["neurons"]:
        position = neuron["position"]
        id = neuron['id']
        x = position[0]
        y = position[1]
        rows.append({"id": id, "x": x, "y": y})

    points = pd.DataFrame(rows)

    num_classes = len(np.unique(y_train))


    for a in data["model"]["neurons"]:
        a["hits"] = np.array(np.zeros(num_classes))

    df = pd.DataFrame(data["model"]["neurons"])
    return df

def getHits(X,y,df):

    num_samples = len(X)
    num_weights = len(df.at[0,"position"])
    num_neurons = len(df)
    input_width = X.shape[1]
    num_classes = len(np.unique(y)) 


    # For all samples
    for s in range(0,num_samples):
        # get sample position
        sample_pos = X[s]
        # init neuron dist arr
        dist = np.zeros(num_neurons)

        # For all Neurons
        for i,row in df.iterrows():
            # get neuron position
            neuron_pos = row["position"]
            # init distance val
            dist_val = 0.0

            # For all neuron weights (all dimensions)
            for a in range(0,num_classes):
                # add squared distances
                diff = sample_pos[a]-neuron_pos[a]
                dist_val += diff * diff

            dist[i] = np.sqrt(dist_val)

        # find best neuron (winner)
        best_neuron_idx = np.argmin(dist)
        sample_class = y[s]

        hits = df.at[best_neuron_idx,"hits"]
        hits[sample_class] +=1

    return df
    
def getHits(X,y,df):

    num_samples = len(X)
    num_weights = len(df.at[0,"position"])
    num_neurons = len(df)
    input_width = X.shape[1]
    num_classes = len(np.unique(y)) 


    # For all samples
    for s in range(0,num_samples):
        # get sample position
        sample_pos = X[s]
        # init neuron dist arr
        dist = np.zeros(num_neurons)

        # For all Neurons
        for i,row in df.iterrows():
            # get neuron position
            neuron_pos = row["position"]
            # init distance val
            dist_val = 0.0

            # For all neuron weights (all dimensions)
            for a in range(0,num_classes):
                # add squared distances
                diff = sample_pos[a]-neuron_pos[a]
                dist_val += diff * diff

            dist[i] = np.sqrt(dist_val)

        # find best neuron (winner)
        best_neuron_idx = np.argmin(dist)
        sample_class = y[s]

        hits = df.at[best_neuron_idx,"hits"]
        hits[sample_class] +=1

    return df

def detect(best_neurons,y,df):
    num_classes = len(np.unique(y)) 
    # -------------------------------------------------
    # Assign class labels to samples using top-k neurons
    # -------------------------------------------------
    num_samples = len(y)
    k = 3  # Use top 3 nearest neurons
    sample_predictions = []

    for s in range(num_samples):
        # Get top k neurons for this sample
        top_k_neurons = best_neurons[best_neurons["sample_idx"] == s].head(k)

        # Initialize class probabilities
        class_probs = np.zeros(num_classes)

        # For each of the top k neurons
        for idx, row in top_k_neurons.iterrows():
            neuron_id = row["neuron_id"]
            rank = row["rank"]

            # Get neuron hits
            neuron_row = df[df["id"] == neuron_id]
            hits = np.array(neuron_row.iloc[0]["hits"])
            total_hits = np.sum(hits)

            # Calculate weight based on rank (lower rank = higher weight)
            weight = 1.0 / (rank + 1)

            # Add weighted class probabilities
            if total_hits > 0:
                class_probs += (hits / total_hits) * weight
            else:
                # If no hits, distribute uniformly
                class_probs += (np.ones(num_classes) / num_classes) * weight

        # Normalize probabilities
        class_probs = class_probs / np.sum(class_probs)

        # Predict class with highest probability
        predicted_class = np.argmax(class_probs)

        sample_predictions.append({
            "sample_idx": s,
            "predicted_class": predicted_class,
            "class_probs": class_probs,
            "true_class": y[s]
        })


    predictions_df = pd.DataFrame(sample_predictions)

    return predictions_df

def findBMU(X_test,gng_res):
    # -------------------------------------------------
    # Inference: Find best matching neurons for X_test
    # -------------------------------------------------
    num_samples_test = len(X_test)
    num_neurons = len(gng_res)
  #  num_classes = len(np.unique(y_train))
    input_width = X_test.shape[1]
    
    # Store results: list of tuples (sannmple_idx, neuron_idx, distance, neuron_id)
    best_neurons = []
    
    # For all test samples
    for s in range(num_samples_test):
        # get sample position
        sample_pos = X_test[s]
        # init neuron dist arr
        dist = np.zeros(num_neurons)
    
        # For all Neurons
        for i, row in gng_res.iterrows():
            # get neuron position
            neuron_pos = row["position"]
           # print(neuron_pos)
            # init distance val
            dist_val = 0.0
    
            # For all neuron weights (all dimensions)
            for a in range(input_width):
                # add squared distances
                diff = sample_pos[a] - neuron_pos[a]
                dist_val += diff * diff
    
            # neuron dist val arr[curr neuron] = sqrt of dist val
            dist[i] = np.sqrt(dist_val)
    
        # Sort neurons by distance (smallest fi
    
    #    print(dist)
        sorted_indices = np.argsort(dist)
    #    print(sorted_indices)
        
        # Store all neurons for this sample, sorted by distance
        for rank, neuron_idx in enumerate(sorted_indices):
            neuron_id = gng_res.at[neuron_idx, "id"]
            distance = dist[neuron_idx]
            best_neurons.append({
                "sample_idx": s,
                "rank": rank,
                "neuron_id": neuron_id,
                "distance": distance
            })
    
    # Convert to DataFrame for easy viewing
    best_neurons_df = pd.DataFrame(best_neurons)
    return best_neurons_df

def calc_classes_numpy(f_points, f_edges):
    # Extract columns as NumPy arrays
    point_ids = f_points["id"].to_numpy()
    point_class = f_points["class"].to_numpy()

    edge_start = f_edges["start"].to_numpy()
    edge_to    = f_edges["to"].to_numpy()

    num_points = len(point_ids)
    num_edges  = len(edge_start)

    # --- Mapping: point_id -> index ---
    # (Assuming ids are arbitrary integers)
    # Build fast lookup dictionary: id → index
    id_to_index = {pid: i for i, pid in enumerate(point_ids)}

    changed = True
    while changed:
        changed = False

        for p_idx in range(num_points):
            curr_point_id = point_ids[p_idx]

            # Loop edges
            for e in range(num_edges):

                s = edge_start[e]
                t = edge_to[e]

                # Case 1: point is edge start
                if s == curr_point_id:
                    i_s = id_to_index[s]
                    i_t = id_to_index[t]

                    if point_class[i_s] > point_class[i_t]:
                        point_class[i_s] = point_class[i_t]
                        changed = True
                        break

                # Case 2: point is edge end
                if t == curr_point_id:
                    i_s = id_to_index[t]
                    i_t = id_to_index[s]

                    if point_class[i_s] > point_class[i_t]:
                        point_class[i_s] = point_class[i_t]
                        changed = True
                        break

    # Write classes back to DataFrame
    f_points["class"] = point_class
    return f_points