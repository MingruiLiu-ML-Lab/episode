function figure2 = objective_plot(Alocal, solutions, plot_path, legend_ind)


    %x_mini = solutions(1);
    x_local = solutions(1);
    x_episode = solutions(2);

    % global minimum
    a = mean(Alocal, 1);
    fun=@(x) a(1)*x.^4 + a(2)*x.^3 + a(3)*x.^2 + a(4)*x;
    if a(2) > 0
        val=fminsearch(fun, -1);
        min=fun(val);
    else
        val=fminsearch(fun, 1);
        min=fun(val);
    end
    % local objectives
    a1 = Alocal(1, :);
    fun1 = @(x) a1(1)*x.^4 + a1(2)*x.^3 + a1(3)*x.^2 + a1(4)*x;

    a2 = Alocal(2, :);
    fun2 = @(x) a2(1)*x.^4 + a2(2)*x.^3 + a2(3)*x.^2 + a2(4)*x;

    % global objective
    figure2 = figure();
    axes1 = axes('Parent',figure2);
    figure2.Position = [200 200 500 450];
    x = linspace(1, 4, 100);
    y = fun(x);
    y1 = fun1(x);
    y2 = fun2(x);
    hold on;
    box on;
    p(1) = plot(x, y', 'LineWidth', 3, 'Color', 'black');
    p(2) = plot(x, y1', '--', 'LineWidth', 3, 'Color', 'black');
    p(3) = plot(x, y2', '-.', 'LineWidth', 3, 'Color', 'black');
    %p(4) = plot(x_mini, fun(x_mini), 'o', 'Color', 'blue', 'MarkerSize', 15,'LineWidth', 2);
    p(4) = plot(x_local, fun(x_local), 'b^', 'MarkerSize', 30, 'LineWidth', 2); 
    p(5) = plot(x_episode, fun(x_episode), 'r*', 'MarkerSize', 30, 'LineWidth', 2);
    set(gca, 'LineWidth', 2);

%     yl = yline(min, ':', 'min $f(x)$', 'Interpreter','latex', 'LineWidth', 1.5);
%     yl.LabelHorizontalAlignment = 'left';
%     yl.LabelVerticalAlignment = 'bottom';
%     yl.Color = 'magenta';
    set(axes1,'FontSize',20,'LineWidth', 1,'XMinorTick','on','YMinorTick','on');
    xlabel({'x'}, 'FontSize', 20);
    ylabel({'Objective value'}, 'FontSize', 20);
    if(legend_ind == 1)
    legend(p, {'$f(x)$', '$f_1(x)$', '$f_2(x)$', 'CELGC', 'EPISODE'},'Interpreter', 'latex', "FontSize", 20, 'Location', 'northwest');
    end

    exportgraphics(figure2, plot_path);
end