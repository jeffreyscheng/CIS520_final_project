function features = sparse_PCA(X, k)
    [U,S,V] = svds(X, k);
    features = U * S;
end