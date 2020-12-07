function ll = Spiking_LL(X, k, Spikes, f)
    
    dt = 200;
    total_t = 200000;

    ll = 0;
    for t = 1:total_t
        lambda_t = f(dot(k, X(:,t))); % don't add b because it's already concatenated within k
        yt = Spikes(t);
        %ll = ll + (yt * log(dt) + yt*log(lambda_t) - dt * lambda_t - log(factorial(yt)));
        ll = ll + (yt * log(lambda_t * dt) - dt * lambda_t - log(factorial(yt)));

    end
    %L(j) = ll;
end