function param_grid = generate_parameter_grid(params)
% Generate an array of parameters from params.
ref_grid = create_ref_grid(params);
param_grid = update_grid_recursive(ref_grid, params);
end


function grid = update_grid_recursive(grid, params)
if isempty(fieldnames(params))
    return;
end

names = fieldnames(params);
name = names{1};
values = getfield(params, name);
cnt = 1;
% skip string type
if ischar(values)
    new_grid = grid;
else
    for i = 1:numel(values)
        for j = 1:numel(grid)
            new_grid(cnt) = setfield(grid(j), name, values(i));
            cnt = cnt + 1;
        end
    end
end
params = rmfield(params, name);
grid = update_grid_recursive(new_grid, params);
end


function ref_grid = create_ref_grid(params)
% Generate a reference grid.
ref_grid = params;
names = fieldnames(params);
for i = 1:numel(names)
    name = names{i};
    values = getfield(params, name);
    % skip string type
    if ischar(values)
        continue;
    end
    ref_grid = setfield(ref_grid, name, values(1));
end
end