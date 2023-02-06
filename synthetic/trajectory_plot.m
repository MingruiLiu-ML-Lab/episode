function figure1 = trajectory_plot(loss_matrix, plot_path, legend_ind)
    % trajectory of loss
    % loss_matrix = [loss_mini loss_local loss_episode]

    n = length(loss_matrix);
    figure1 = figure();
    figure1.Position = [10 10 500 450];
    axes1 = axes('Parent',figure1);
    plot1 = plot(0:(n-1), loss_matrix, 'LineWidth', 3);
    %set(plot1(1),'DisplayName','Minibatch Clipping','Color',[0 0 1]);
    set(plot1(1),'DisplayName','CELGC', 'Color',[0 0 1]);
    set(plot1(2),'DisplayName','EPISODE', 'Color', [1 0 0]);
    set(gca, 'LineWidth', 2);
    xlabel('Round', 'FontSize', 20);
    ylabel({'Objective value'}, 'FontSize', 20);
    
    set(axes1,'FontSize',20,'LineWidth',1,'XMinorTick','on','YMinorTick','on');
    if(legend_ind == 1)
        legend("FontSize", 20, 'Location', 'Best');
    end

    exportgraphics(figure1, plot_path);
end