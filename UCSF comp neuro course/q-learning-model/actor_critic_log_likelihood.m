function log_likelihood = actor_critic_log_likelihood(choices, outcomes, w1, w2, v, a_w, a_v, b)

    % arrays
    num_trials = length(choices);
    ws_1 = zeros(1, num_trials+1);
    ws_2 = zeros(1, num_trials+1);
    state_vals = zeros(1, num_trials+1);
    prob_choice = zeros(1, num_trials);
    
    % fill out
    ws_1(1) = w1;
    ws_2(1) = w2;
    state_vals(1) = v;
    for t = 1:num_trials
       
        % local ws
        w1 = ws_1(t);
        w2 = ws_2(t);
        v = state_vals(t);
        prob_1 = 1 / (1 + exp(-b * (w1 - w2)));
        
        % calculate choice and rpe
        choice = choices(t);
        reward = outcomes(t);
        
        % calculate rpe and new v
        rpe = reward - v;
        if choice == 1
            
            % add probability from choosing
            prob_choice(t) = prob_1;
            w1 = w1 + (a_w * rpe);
        else
            prob_choice(t) = 1 - prob_1;
            w2 = w2 + (a_w * rpe);
        end
        
        % update vals for next iteration
        state_vals(t+1) = v + (a_v * rpe);
        ws_1(t+1) = w1;
        ws_2(t+1) = w2;
        
    end

    % calculate log likelihood
    log_likelihood = log(prod(prob_choice));

end