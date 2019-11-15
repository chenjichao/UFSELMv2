function plot_acc_bar(param_grid, acc_mat, varx, vary)
% Plot the 3d errorbar corresponding to varx and vary.
gridx = grid_points(param_grid, varx);
gridy = grid_points(param_grid, vary);
disp(varx);
disp(gridx);
disp(vary);
disp(gridy);
assert(numel(gridx)*numel(gridy) == numel(param_grid));
cmapx = containers.Map(gridx, 1:numel(gridx));
cmapy = containers.Map(gridy, 1:numel(gridy));
grid = zeros(numel(gridy), numel(gridx));
max_acc = -1;
for i = 1:numel(param_grid)
    x = cmapx(param_grid(i).(varx));
    y = cmapy(param_grid(i).(vary));
    accs = zeros(size(acc_mat, 1), 1);
    for n = 1:size(acc_mat, 1)
        accs(n) = last_nonzero(vec(acc_mat(n, i, :)));
    end
    grid(y, x) = mean(accs);
    if grid(y, x) > max_acc
        max_acc = grid(y, x);
        max_loc = [y, x];
    end
end

% plot bars and highlight the best one in red
h = bar3(grid);
assert(length(h) == length(gridx));
cm = get(gcf,'colormap');  % Use the current colormap.
cnt = 0;
for jj = 1:length(h)
    xd = get(h(jj),'xdata');
    yd = get(h(jj),'ydata');
    zd = get(h(jj),'zdata');
    delete(h(jj))    
    idx = [0;find(all(isnan(xd),2))];
    if jj == 1
        S = zeros(length(h)*(length(idx)-1),1);
        dv = floor(size(cm,1)/length(h));
    end
    for ii = 1:length(idx)-1
        cnt = cnt + 1;
        if (jj == max_loc(2)) && (ii == max_loc(1))
%             facecolor = 'red';
            facecolor = cm(jj*dv,:);
        else
            facecolor = cm(jj*dv,:);
        end
        S(cnt) = surface(xd(idx(ii)+1:idx(ii+1)-1,:),...
                         yd(idx(ii)+1:idx(ii+1)-1,:),...
                         zd(idx(ii)+1:idx(ii+1)-1,:),...
                         'facecolor',facecolor);
    end
end

% add accuracy values
[xloc, yloc] = meshgrid(1:size(grid, 2), 1:size(grid, 1));
% text(xloc(:), yloc(:), grid(:), num2str(grid(:), '%.4f'), ...
%     'HorizontalAlignment', 'center', ...
%     'VerticalAlignment', 'bottom');
xlabel('log_{10}\lambda_2');
ylabel('log_{10}\lambda_1');
zlim([0 1]);
% xlabel(varx);
% ylabel(vary);
% zlabel('accuracy');
xticklabels(split(num2str(gridx)));
yticklabels(split(num2str(gridy)));
end


function p = grid_points(param_grid, var)
all_vals = [];
for i = 1:numel(param_grid)
    all_vals = [all_vals, param_grid(i).(var)];
end
p = sort(unique(all_vals));
end

function out = last_nonzero(x)
% Returns the last nonzero elements in x.
idx = sort(find(x ~= 0));
if isempty(idx)
    out = 0;
else
    out = x(idx(end));
end
end