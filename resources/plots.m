
%% Supervised
clear;clc;close all;
accuracy_test_mlp = csvread('run_sl_mlp-tag-accuracy_test.csv',1);
accuracy_train_mlp = csvread('run_sl_mlp-tag-accuracy_train.csv',1);

accuracy_test_cnn = csvread('run_sl_cnn-tag-accuracy_test.csv',1);
accuracy_train_cnn = csvread('run_sl_cnn-tag-accuracy_train.csv',1);

loss_error_cnn = csvread('run_sl_cnn-tag-loss_error (1).csv',1);
loss_error_mlp = csvread('run_sl_mlp-tag-loss_error (1).csv',1);

figure, hold on, grid on;
plot(accuracy_test_cnn(:,2), smooth(accuracy_test_cnn(:,3)), 'Color', [0.9336    0.3242    0.3125], 'DisplayName', 'Testing accuracy');
plot(accuracy_train_cnn(:,2), smooth(accuracy_train_cnn(:,3)), 'Color', [0.2578    0.6445    0.9570], 'DisplayName', 'Training accuracy');
axis([0 15e4, 0.85, 1.025]);
title('Training vs Testing Accuracies for CNN', 'FontSize', 16);
lgd = legend('show');
lgd.FontSize = 12;
xlabel('Epochs', 'FontSize', 14);
ylabel('Accuracy', 'FontSize', 14);

figure, hold on, grid on;
plot(accuracy_test_mlp(:,2), smooth(accuracy_test_mlp(:,3)), 'Color', [0.9336    0.3242    0.3125], 'DisplayName', 'Testing accuracy');
plot(accuracy_train_mlp(:,2), smooth(accuracy_train_mlp(:,3)), 'Color', [0.2578    0.6445    0.9570], 'DisplayName', 'Training accuracy');
axis([0 15e4, 0.85, 1.025]);
title('Training vs Testing Accuracies for MLP', 'FontSize', 16);
lgd = legend('show');
lgd.FontSize = 12;
xlabel('Epochs', 'FontSize', 14);
ylabel('Accuracy', 'FontSize', 14);

figure, hold on, grid on;
plot(loss_error_cnn(:,2), smooth(loss_error_cnn(:,3)), 'Color', [0.9336    0.3242    0.3125], 'DisplayName', 'CNN');
plot(loss_error_mlp(:,2), smooth(loss_error_mlp(:,3)), 'Color', [0.2578    0.6445    0.9570], 'DisplayName', 'MLP');
axis([0 16e4, 0, 0.5]);
title('Cross-Entropy Loss Function', 'FontSize', 16);
lgd = legend('show');
lgd.FontSize = 12;
xlabel('Epochs', 'FontSize', 14);
ylabel('Loss', 'FontSize', 14);
%% Benchmarking 1
clear; clc; close all;

mlp_draws = csvread('run_rl_mlp_multi-tag-draws_random.csv', 1)
mlp_losses = csvread('run_rl_mlp_multi-tag-losses_random.csv', 1)
cnn_draws = csvread('run_rl_cnn_multi-tag-draws_random.csv', 1)
cnn_losses = csvread('run_rl_cnn_multi-tag-losses_random.csv', 1)

%Loss
figure, hold on, grid on;
plot(mlp_losses(:,2), mlp_losses(:,3), 'Color', [0.9336    0.3242    0.3125], 'DisplayName', 'MLP', 'LineWidth', 1.8);
plot(cnn_losses(:,2), cnn_losses(:,3), 'Color', [0.2578    0.6445    0.9570], 'DisplayName', 'CNN', 'LineWidth', 1.8);
set(gca,'fontsize',14)
% axis([0 15e4, 0.85, 1.025]);
title('Loss Outcome Rate', 'FontSize', 16);
lgd = legend('show');
lgd.FontSize = 14;
xlabel('Epochs', 'FontSize', 16);
ylabel('Count', 'FontSize', 16);

%Draws
figure, hold on, grid on;
plot(mlp_draws(:,2), mlp_draws(:,3), 'Color', [0.9336    0.3242    0.3125], 'DisplayName', 'MLP', 'LineWidth', 1.8);
plot(cnn_draws(:,2), cnn_draws(:,3), 'Color', [0.2578    0.6445    0.9570], 'DisplayName', 'CNN', 'LineWidth', 1.8);
set(gca,'fontsize',14)
% axis([0 15e4, 0.85, 1.025]);
title('Draw Outcome Rate', 'FontSize', 16);
lgd = legend('show');
lgd.FontSize = 14;
xlabel('Epochs', 'FontSize', 16);
ylabel('Count', 'FontSize', 16);

%% Benchmarking 2
clear; clc; close all;
mlp_deproved_random_reinforcement = csvread('run_rl_mlp_final_benchmark_long-tag-deproved_random.csv',1);
mlp_improved_random_reinforcement = csvread('run_rl_mlp_final_benchmark_long-tag-improved_random.csv',1);
mlp_deproved_random_supervised = csvread('run_sl_mlp-tag-deproved_random.csv',1);
mlp_improved_random_supervised = csvread('run_sl_mlp-tag-improved_random.csv',1);



% mlp for sl (same as above)

figure, hold on, grid on;
plot(mlp_deproved_random_reinforcement(:,2), mlp_deproved_random_reinforcement(:,3), 'Color', [0.2578    0.6445    0.9570], 'DisplayName', 'Reinforcemet deproved rate', 'LineWidth', 1.8)
plot(mlp_improved_random_reinforcement(:,2), mlp_improved_random_reinforcement(:,3), 'Color', [0.9336    0.3242    0.3125], 'DisplayName', 'Reinforcemet improved rate', 'LineWidth', 1.8)
plot(mlp_deproved_random_supervised(:,2), mlp_deproved_random_supervised(:,3), '--', 'Color', [0.2578    0.6445    0.9570], 'DisplayName', 'Supervised deproved rate', 'LineWidth', 1.8);
plot(mlp_improved_random_supervised(:,2), mlp_improved_random_supervised(:,3), '--', 'Color', [0.9336    0.3242    0.3125], 'DisplayName', 'Supervised deproved rate', 'LineWidth', 1.8);
axis([0,3.7e04,0, 0.9]);
title('Deprovement vs Improvement Rates, MLP against Random Agent', 'FontSize', 16);
lgd = legend('show');
lgd.FontSize = 12;
xlabel('Epochs', 'FontSize', 14);
ylabel('%', 'FontSize', 14);