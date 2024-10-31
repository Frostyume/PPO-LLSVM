function K = rbf_kernel(X1, X2, gamma)
    sqX1 = sum(X1.^2, 2);
    sqX2 = sum(X2.^2, 2);
    K = exp(-gamma * (bsxfun(@plus, sqX1, sqX2') - 2 * (X1 * X2')));
end
