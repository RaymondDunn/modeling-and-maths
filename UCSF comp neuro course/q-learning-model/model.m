%% part 1
%% prepare simulation
% load probabilities
prob_rwd_1 = csvread('prob_rwd_1.csv');
prob_rwd_2 = csvread('prob_rwd_2.csv');

% initialize
num_trials = 361;
a = 0.2;
b = 3;

% vars for qs/outputs
%rand_nums = rand(1, num_trials);
qs_1 = zeros(1, num_trials + 1);
qs_2 = zeros(1, num_trials + 1);
qs_1(1) = 0.5;
qs_2(1) = 0.5;
prob_choice_1 = zeros(1, num_trials);
prob_choice_2 = zeros(1, num_trials);
rewarded = zeros(1, num_trials);

%% run simulation
% iterate trials
for i = 1:num_trials
    
    % get qs
    q1 = qs_1(i);
    q2 = qs_2(i);
    
    % calculate probabilities of choosing
    prob_1 = 1 / (1 + exp(-b *(q1-q2)));
    prob_2 = 1 - prob_1;
    
    % save prob of choice
    prob_choice_1(i) = prob_1;
    prob_choice_2(i) = prob_2;
    
    % get p_reward and update qs if necessary
    rand_num = rand();
    if rand_num <= prob_1
        p_reward = prob_rwd_1(i);
        reward = rand_num <= p_reward;
        rpe = reward - q1;
        new_q = q1 + (a * rpe);
        qs_1(i+1) = new_q;
        qs_2(i+1) = q2;
    else
        p_reward = prob_rwd_2(i);
        reward = rand_num <= p_reward;
        rpe = reward - q2;
        new_q = q2 + (a * rpe);
        qs_1(i+1) = q1;
        qs_2(i+1) = new_q;
    end
    
    % store if we got a reward
    rewarded(i) = reward;
end

% trim last trial..
qs_1 = qs_1(1:num_trials);
qs_2 = qs_2(1:num_trials);

%% plot results of simulation
figure
hold on
plot(prob_rwd_1)
plot(qs_1)
plot(prob_choice_1)
title('qs for 1')
legend('prob rwd', 'qs', 'prob choice 1')

figure
hold on
plot(prob_rwd_2)
plot(qs_2)
plot(prob_choice_2)
title('qs for 2')
legend('prob rwd', 'qs', 'prob choice 2')

%% compute likelihood model produces this sequence of outcomes given the 
% reward for each choice
% take qs and action values to compute probability
% calculate the likelihood of the sequence of choices given the model and
% outcomes
trials_chosen_1 = prob_choice_1 >= prob_choice_2;
trials_chosen_2 = prob_choice_1 < prob_choice_2;
probs_chosen = prob_choice_1 .* trials_chosen_1 + prob_choice_2 .* trials_chosen_2;
ll = log(prod(probs_chosen));

%% calculate log likelihood w/ function
% compare to 1b
choices = trials_chosen_1 + trials_chosen_2 * 2;
outcomes = rewarded;
ll_new = q_log_likelihood(choices, outcomes, 0.5, 0.5, 0.2, 3);


%% fit model
lol = fmincon(@(x) -q_log_likelihood(choices, outcomes, 0.5, 0.5, x(1), x(2)), [0.1, 4]);


%% plot surface
alphas = linspace(0.01, 0.6);
betas = linspace(0.1, 6);
ll_mat = zeros(length(alphas), length(betas));

% build surface matrix
for a = 1:length(alphas)
    for b = 1:length(betas)
        ll_mat(a, b) = -q_log_likelihood(choices, outcomes, 0.5, 0.5, alphas(a), betas(b));
    end
end

% plot
figure()
surf(ll_mat)
xlabel('alphas');
ylabel('betas');
xticks = alphas;
yticks = betas;

%% part 2
%% setup for simulation
% vectors
ws_1 = zeros(1, num_trials + 1);
ws_2 = zeros(1, num_trials + 1);
rand_nums = rand(1, num_trials);
state_vals = zeros(1, num_trials + 1);
choices = zeros(1, num_trials);

% initial vals
a_w = 0.5;
a_v = 0.5;
b = 3;
num_trials = 361;
ws_1(1) = 0.5;
ws_2(1) = 0.5;
state_vals(1) = 0.4;

%% run simulation
for i = 1:num_trials
    
    % local vars
    w1 = ws_1(i);
    w2 = ws_2(i);
    v = state_vals(i);
    
    % calculate probability
    prob_1 = 1/(1+exp(-b * (w1 - w2)));
    prob_2 = 1 - prob_1;

    % make a choice
    rand_num = rand();
    if rand_num <= prob_1
        reward = rand() < prob_rwd_1(i);
        rpe = reward - v;
        
        % update w
        w1 = w1 + (a_w * rpe);
        
        % add choice/rewarded
        choices(i) = 1;
        rewarded(i) = reward;
    else
        reward = rand() < prob_rwd_2(i);
        rpe = reward - v;
        
        % update w 
        w2 = w2 + (a_w * rpe);
        
        % add choice/rewarded
        choices(i) = 2;
        rewarded(i) = reward;
        
    end
    
    % update state vals
    v = v + (a_v * rpe);
    
    % store next state
    ws_1(i+1) = w1;
    ws_2(i+1) = w2;
    state_vals(i+1) = v;
end

figure
hold on
plot(ws_1)
plot(prob_rwd_1)
plot(state_vals)
legend('w_1', 'prob rwd_1', 'state vals')
xlabel('trials')

figure
hold on
plot(ws_2)
plot(prob_rwd_2)
plot(state_vals)
legend('w_2', 'prob rwd_2', 'state vals')
xlabel('trials')
% plot for choice 1
%figure
%hold on
%plot(choices)
%plot(rewarded)

%% calculate log likelihood for actor-critic
ll = actor_critic_log_likelihood(choices, outcomes, ws_1(1), ws_2(1), state_vals(1), a_w, a_v, b);

%% fit model
lol = fmincon(@(x) -actor_critic_log_likelihood(choices, outcomes, ws_1(1), ws_2(1), state_vals(1), x(1), x(2), x(3)), [0.4, 0.3, 1], [], [], [], [], [0, 0, 0], [1, 1, 10]);

%% plot surface
a_ws = linspace(0.001, 1);
a_vs = linspace(0.001, 1);
betas = linspace(0.5, 10);
ll_mat = zeros(length(alphas), length(betas));

% fix a_w
a_w = 0.5;
for i = 1:length(a_vs)
    for j = 1:length(betas)
        ll_mat(i, j) = -actor_critic_log_likelihood(choices, outcomes, ws_1(1), ws_2(1), state_vals(1), a_w, a_vs(i), betas(j));
    end
end
figure
surf(ll_mat)
xlabel('a_v')
ylabel('betas')

% fix a_v
a_v = 0.5;
for i = 1:length(a_ws)
    for j = 1:length(betas)
        ll_mat(i, j) = -actor_critic_log_likelihood(choices, outcomes, ws_1(1), ws_2(1), state_vals(1), a_ws(i), a_v, betas(j));
    end
end
figure
surf(ll_mat)
xlabel('a_w')
ylabel('betas')

% fix beta
b = 3;
for i = 1:length(a_ws)
    for j = 1:length(a_vs)
        ll_mat(i, j) = -actor_critic_log_likelihood(choices, outcomes, ws_1(1), ws_2(1), state_vals(1), a_ws(i), a_vs(j), b);

    end
end
figure
surf(ll_mat)
xlabel('a_w')
ylabel('a_v')
