function [trainmse, trainvarratio] = give_me_measure(y_hat_grad_train, ytrain, vartrain)
    trainmse = sum((y_hat_grad_train - ytrain) .^ 2) / size(ytrain, 1);
    trainvarratio = trainmse / vartrain;
end