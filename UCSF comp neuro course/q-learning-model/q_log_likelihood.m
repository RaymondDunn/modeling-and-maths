function log_likelihood = q_log_likelihood(choices, outcomes, q1, q2, a, b)
    
    num_trials = length(choices);
    qs_1 = zeros(1, num_trials+1);
    qs_2 = zeros(1, num_trials+1);
    qs_1(1) = q1;
    qs_2(1) = q2;
    prob_choice = zeros(1, num_trials);
    
    % choice will update action value based on the outcome
    for t = 1:num_trials
        
        % local qs
        q1 = qs_1(t);
        q2 = qs_2(t);
        prob_1 = 1 / (1 + exp(-1 * b * (q1 - q2)));
        
        % calculate choice and rpe
        choice = choices(t);
        reward = outcomes(t);
        if choice == 1
            rpe = reward - q1;
            new_q = q1 + (a * rpe);
            prob_choice(t) = prob_1;
            qs_1(t+1) = new_q;
            qs_2(t+1) = q2;
        else
            rpe = reward - q2;
            new_q = q2 + (a * rpe);
            prob_choice(t) = 1 - prob_1;
            qs_1(t+1) = q1;
            qs_2(t+1) = new_q;
        end
    end
    
    log_likelihood = log(prod(prob_choice));

    

    
end