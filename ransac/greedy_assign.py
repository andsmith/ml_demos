import numpy as np
import matplotlib.pyplot as plt

def greedy_assign(distances, max_dist):
    """
    Assign correspondences between two sets of points using a greedy algorithm.
    Assign each column-point to the closest row-point, removing each from consideration until remaining distances are too large.

    :param distances: M x N distances between M sources and N targets
    :param max_dist: maximum distance between a source and a target for an assignment to be valid
    :return: list of tuples, where each tuple is an assignment of a source to a target
             list of unassigned columns
             list of unassigned rows
    """
    M, N = distances.shape
    big_dist = (np.max(distances) + max_dist) * 2.0  # big distance to use for already matched corners
    assigned_a = np.zeros(M, dtype=bool)  # assigned_2[i]=False if row i is not assigned to any column
    assigned_b = np.zeros(N, dtype=bool)  # assigned_2[j]=False if column j is not assigned to any row
    assignments = []

    # Assign each true corner to the closest detected corner and
    #    remove the true corner from the unassigned list.
    # Break when the distance is less than max_dist_px.
    for i in range(M):
        
        dist_row = distances[i] + big_dist * assigned_b  # don't match this true corner to any already matched corner
        j = np.argmin(dist_row)
        if dist_row[j] < max_dist:
            assigned_a[i] = True
            assigned_b[j] = True
            assignments.append((j, i))

    unassigned_a = np.where(np.logical_not(assigned_a))[0]
    unassigned_b = np.where(np.logical_not(assigned_b))[0]

    return assignments, unassigned_a, unassigned_b



def test_greedy_assign(plot = False):
    """
    Test the greedy assignment algorithm.
    set 1 = random points
    set 2 = [set 1 points + small noise] + [additional random points]
    plot results
    """
    np.random.seed(0)
    n_pts = 20
    noise_sigma = 0.001
    max_dist = 0.1

    # create points
    pts1 = np.random.rand(n_pts, 2)
    # add noise to points
    pts2 = pts1 + np.random.randn(n_pts, 2) * noise_sigma
    # add additional points
    pts2 = np.vstack((pts2, np.random.rand(n_pts, 2)))

    # compute distances between points
    distances = np.linalg.norm(pts1[:, None] - pts2, axis=2)

    # run greedy assignment
    assignments, unassigned_a, unassigned_b = greedy_assign(distances, max_dist)
    assigned_a = np.array([a for a, b in assignments])
    assigned_b = np.array([b for a, b in assignments])

    # check correct points are assigned correctly
    assert len(assignments) == n_pts, "Number of points assinged is wrong:  %i != %i" % (len(assignments), n_pts)
    for a, b in assignments:
        assert np.linalg.norm(pts1[a] - pts2[b]) < max_dist, "Point %i is not assigned to point %i" % (a, b)
    for a in unassigned_a:
        assert a not in assigned_a, "Point %i is unassigned but should not be" % a
    for b in unassigned_b:
        assert b not in assigned_b, "Point %i is unassigned but should not be" % b


    if plot:
        plt.figure()
        plt.plot(pts1[:, 0], pts1[:, 1], 'rx', label='set 1')
        plt.plot(pts2[:, 0], pts2[:, 1], 'bo', label='set 2')
        for a, b in assignments:
            plt.plot([pts1[a, 0], pts2[b, 0]], [pts1[a, 1], pts2[b, 1]], 'k-')
        plt.title('Greedy assignment')
        plt.legend()
        plt.show()

    return assignments, unassigned_a, unassigned_b

    
        
if __name__ == '__main__':
    test_greedy_assign()
    print("Greedy assignment test passed")