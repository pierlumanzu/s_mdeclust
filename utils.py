import networkx as nx


def create_groups(D, ML, CL):

    ML_groups = [concom for concom in nx.connected_components(nx.from_edgelist(ML))]

    for idx_p in range(len(D)):
        already_added = False
        for ml_gr in ML_groups:
            if idx_p in ml_gr:
                already_added = True
                break
        
        if not already_added:
            ML_groups.append({idx_p})

    CL_groups = []
    for _ in ML_groups:
        CL_groups.append([])

    for cl in CL:

        found_0 = False
        found_1 = False

        for idx_ml_gr in range(len(ML_groups)):
            
            if cl[0] in ML_groups[idx_ml_gr]:
                found_0 = True
                idx_0 = idx_ml_gr
            
            if cl[1] in ML_groups[idx_ml_gr]:
                found_1 = True
                idx_1 = idx_ml_gr

            if found_0 and found_1:
                break

        assert idx_0 != idx_1

        CL_groups[idx_0].append(idx_1)
        CL_groups[idx_1].append(idx_0)

    for idx_cl_gr in range(len(CL_groups)):
        CL_groups[idx_cl_gr] = set(CL_groups[idx_cl_gr])

    return ML_groups, CL_groups