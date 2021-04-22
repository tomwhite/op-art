// Uncomment following when changing js and developing in jupyter
// require.undef("op_art");

define("op_art", ["d3"], function (d3) {

  /*
   * Test if an object is a string.
   */
  function isString(s) {
    return typeof s === "string" || s instanceof String;
  }

  /*
   * Format a value of any type (boolean, number, string) as a string for display.
   */
  function formatValue(v) {
    if (typeof v == "boolean") {
      return v ? "True" : "False";
    } else if (typeof v == "number") {
      if (Number.isInteger(v)) {
        return v;
      }
      return d3.format(",.2g")(v);
    }
    return v;
  }

  /*
   * Format an index.
   */
  function formatIndex(i) {
    if (Array.isArray(i)) {
      return `(${i.join(", ")})`
    }
    return i; // number
  }

  /*
   * Return the elements (cells/chunks) of an array.
   */
  function arrayElements(arr) {
    // array-api uses cells, dask uses chunk_reps
    return arr.cells || arr.chunk_reps;
  }

  /*
   * Return the extent (minimum and maximum) values across all arrays.
   */
  function valuesExtent(arrs) {
    const allCells = arrs.map((o) => arrayElements(o)).flat(1);
    if (typeof allCells[0].value === 'undefined') { // dask arrays are lazy (no value), so just return unit interval
      return [0, 1]
    }
    return d3.extent(allCells, (d) => d.value); // array-api case
  }

  /*
   * Return true if arr has no input sources.
   */
  function noSources(arr) {
    return arrayElements(arr).every((elt) => !elt.sources);
  }

  /*
   * Return true if arr should have per-cell animation, or false if
   * all cells should be animated simulataneously.
   */
  function animatePerCell(arr) {
    // Animate each cell individually only if the output array
    // depends on a large number of inputs. We define that as
    // a cell having more than one source array and more than one cell
    // from a source array.
    if (arrayElements(arr).length == 0) {
      return false;
    }

    const cell = arrayElements(arr)[0]; // use first cell
    const sourceArrays = cell.sources.map((id) => id.split(/_/)[0]);
    const sourceCounts = sourceArrays.reduce(
      (acc, e) => acc.set(e, (acc.get(e) || 0) + 1),
      new Map()
    );
    const numSourceArrays = [...sourceCounts.keys()].length;
    const maxCellsInSourceArray = Math.max(...sourceCounts.values());
    if (numSourceArrays == 1 || maxCellsInSourceArray == 1) {
      return false;
    }
    return true;
  }

  /*
   * Compute the x, y dimensions for drawing an array of ndim<=3.
   */
  function computeArrayDim(arr) {
    const s = arr.shape;
    if (s.length == 0) {
      return [1, 1];
    } else if (s.length == 1) {
      return [s[0], 1];
    } else if (s.length == 2) {
      return [s[1], s[0]];
    } else if (s.length == 3) {
      // TODO: why '- 1' below?
      const stacking = (s[0] - 1) * 0.5;
      return [s[2] + stacking, s[1] + stacking];
    }
  }

  /*
   * Convert an array index to an x value.
   */
  function indexToX(index) {
    if (index.length == 0) {
      return 0;
    } else if (index.length == 1) {
      return index[0];
    } else if (index.length == 2) {
      return index[1];
    } else if (index.length == 3) {
      const stacking = index[0] * 0.5;
      return index[2] + stacking;
    }
  }

  /*
   * Convert an array index to a y value.
   */
  function indexToY(index, shape) {
    if (index.length == 0) {
      return 0;
    } else if (index.length == 1) {
      return 0;
    } else if (index.length == 2) {
      return index[0];
    } else if (index.length == 3) {
      const stacking = (shape[0] - index[0]) * 0.5;
      return index[1] + stacking;
    }
  }

  /*
   * Compute the display width and height of a cell for all given arrays.
   */
  function computeCellDim(container, objects, minCellWidth) {
    if (objects[0].cells) { // array-api
      const valuesAsText = [];
      for (const o of objects) {
        if (!isString(o)) {
          const vals = arrayElements(o).map((d) => formatValue(d.value));
          valuesAsText.push(...vals);
        }
      }
      const max = Math.max(...computeTextWidths(container, valuesAsText));
      const cellWidth = Math.max(minCellWidth, max);
      const cellHeight = minCellWidth;
      return [cellWidth, cellHeight];
    } else { // dask
      const widths = [];
      for (const o of objects) {
        const width = computeArrayDim(arrayElements(o)[0])[0]
        widths.push(width);
      }
      const max = Math.max(...widths);
      const cellWidth = (minCellWidth * 1.0) / max;
      return [cellWidth, cellWidth];
    }
  }

  /*
   * Compute the display widths of each string in textData.
   */
  function computeTextWidths(container, textData) {
    // From https://stackoverflow.com/a/37528489
    const textWidths = [];

    const svgTemp = d3
      .select(container)
      .append("svg")
      .attr("width", 100)
      .attr("height", 40);

    svgTemp
      .append("g")
      .selectAll(".dummyText") // declare a new CSS class 'dummyText'
      .data(textData)
      .enter() // create new element
      .append("text") // add element to class
      .attr("class", "fixed-text")
      .attr("opacity", 0.0) // so it's invisible
      .text((d) => formatValue(d))
      .each(function (d, i) {
        const thisWidth = this.getComputedTextLength();
        textWidths.push(thisWidth);
        this.remove(); // remove them just after displaying them
      });

    svgTemp.remove();

    return textWidths;
  }

  /*
   * Compute the display width of each array in arrays.
   */
  function computeArrayWidths(arrays, cellWidth) {
    return arrays.map((array) => computeArrayDim(array)[0] * cellWidth);
  }

  /*
   * Compute the display height of each array in arrays.
   */
  function computeArrayHeights(arrays, cellHeight) {
    return arrays.map((array) => computeArrayDim(array)[1] * cellHeight);
  }

  /*
   * Compute the display width of each object (array or string) in objects.
   */
  function computeWidths(container, objects, cellWidth) {
    return objects.map((o) => {
      if (isString(o)) {
        return computeTextWidths(container, [o])[0];
      } else {
        return computeArrayWidths([o], cellWidth)[0];
      }
    });
  }

  /*
   * Compute the display height of each object (array or string) in objects.
   */
  function computeHeights(container, objects, cellHeight) {
    return objects.map((o) => {
      if (isString(o)) {
        return 0;
      } else if (!isString(o)) {
        return computeArrayHeights([o], cellHeight)[0];
      }
    });
  }

  function plotCodeAndArray(
    g,
    ndarr,
    codeLine,
    cellWidth,
    cellHeight,
    duration
  ) {

    // Draw line of code
    g.append("g")
      .attr("transform", `translate(${codeLine.x}, ${codeLine.y})`)
      .append("text")
      .attr("class", "fixed-text")
      .text(codeLine.text)
      .attr("x", 0)
      .attr("y", 0);

    // Draw dimension lines
    const x = d3
      .scaleLinear()
      .domain([0, 10])
      .range([0, 10 * cellWidth]);
    const y = d3
      .scaleLinear()
      .domain([0, 10])
      .range([0, 10 * cellHeight]);

    const xOffset = ndarr.x;
    const yOffset = ndarr.y;

    if (ndarr.ndim == 1) {
      const ind1 = [0];
      const ind2 = [ndarr.shape[0]];
      g.append("line")
        .attr("class", "dimension-line")
        .attr("x1", x(indexToX(ind1)) + xOffset)
        .attr("y1", y(indexToY(ind1, ndarr.shape) + 1) + yOffset)
        .attr("x2", x(indexToX(ind2) - 0.1) + xOffset)
        .attr("y2", y(indexToY(ind2, ndarr.shape) + 1) + yOffset);
    } else if (ndarr.ndim == 2) {
      const ind1 = [0, 0];
      const ind2 = [ndarr.shape[0], 0];
      const ind3 = [ndarr.shape[0], ndarr.shape[1]];
      g.append("line")
        .attr("class", "dimension-line")
        .attr("x1", x(indexToX(ind1) - 0.1) + xOffset)
        .attr("y1", y(indexToY(ind1, ndarr.shape)) + yOffset)
        .attr("x2", x(indexToX(ind2) - 0.1) + xOffset)
        .attr("y2", y(indexToY(ind2, ndarr.shape)) + yOffset);
      g.append("line")
        .attr("class", "dimension-line")
        .attr("x1", x(indexToX(ind2) - 0.1) + xOffset)
        .attr("y1", y(indexToY(ind2, ndarr.shape)) + yOffset)
        .attr("x2", x(indexToX(ind3) - 0.1) + xOffset)
        .attr("y2", y(indexToY(ind3, ndarr.shape)) + yOffset);
    } else if (ndarr.ndim == 3) {
      const ind1 = [ndarr.shape[0], 0, 0];
      const ind2 = [0, 0, 0];
      const ind3 = [0, ndarr.shape[1], 0];
      const ind4 = [0, ndarr.shape[1], ndarr.shape[2]];
      g.append("line")
        .attr("class", "dimension-line")
        .attr("x1", x(indexToX(ind1) - 0.1) + xOffset)
        .attr("y1", y(indexToY(ind1, ndarr.shape)) + yOffset)
        .attr("x2", x(indexToX(ind2) - 0.1) + xOffset)
        .attr("y2", y(indexToY(ind2, ndarr.shape)) + yOffset);
      g.append("line")
        .attr("class", "dimension-line")
        .attr("x1", x(indexToX(ind2) - 0.1) + xOffset)
        .attr("y1", y(indexToY(ind2, ndarr.shape)) + yOffset)
        .attr("x2", x(indexToX(ind3) - 0.1) + xOffset)
        .attr("y2", y(indexToY(ind3, ndarr.shape)) + yOffset);
      g.append("line")
        .attr("class", "dimension-line")
        .attr("x1", x(indexToX(ind3) - 0.1) + xOffset)
        .attr("y1", y(indexToY(ind3, ndarr.shape)) + yOffset)
        .attr("x2", x(indexToX(ind4) - 0.1) + xOffset)
        .attr("y2", y(indexToY(ind4, ndarr.shape)) + yOffset);
    }

    plotCells(
      g,
      arrayElements(ndarr),
      duration,
      0
    );
  }

  function plotCells(
    g,
    cells,
    duration,
    delay,
    callback
  ) {
    function frontToBack(a, b) {
      return b.index[0] - a.index[0];
    }
  
    const t = g.transition().delay(delay).duration(duration);
  
    if (typeof cells[0].value !== 'undefined') { // array-api
      const cellElts = g
        .selectAll(".cell")
        .data(cells, (d) => d.key)
        .join(
          (enter) => {
            const gCell = enter
              .append("g")
              .attr("transform", (d) => `translate(${d.x}, ${d.y})`)
              .attr("visibility", "hidden");
            gCell
              .append("rect")
              .attr("class", "cell-rect")
              .attr("x", 0)
              .attr("y", 0)
              .attr("width", d => d.width)
              .attr("height", d => d.height)
              .attr("fill", (d) => d.fill);
            gCell
              .append("text")
              .attr("class", "cell-text")
              .text((d) => d.value)
              .attr("x", 0)
              .attr("y", 0)
              .attr("dx", d => d.width / 2)
              .attr("dy", d => d.height / 2);
            gCell.append("title").text((d) => d.title);
            gCell
              .transition()
              .duration(0)
              .delay(delay)
              .attr("visibility", "visible");
            return gCell;
          },
          (update) =>
            update.call((update) =>
              update
                .transition(t)
                .attr("transform", (d) => `translate(${d.x}, ${d.y})`)
                .end()
                .then(() => {
                  console.log("transition ended");
                  update.remove();
                  callback();
                })
            ),
          (exit) =>
            exit.call((exit) => exit.transition(t).attr("fill", "white").remove()) // fade out
        )
        .attr("class", "cell");
      cellElts.sort(frontToBack);
    } else { // dask
      const cellElts = g
        .selectAll(".cell")
        .data(cells, (d) => d.key)
        .join(
          (enter) => {
            const gCell = enter
              .append("g")
              .append("rect")
              .attr("class", "cell-rect")
              .attr("x", (d) => d.x)
              .attr("y", (d) => d.y)
              .attr("width", (d) => d.width)
              .attr("height", (d) => d.height)
              .attr("fill", (d) => d.fill)
              .attr("visibility", "hidden");
            gCell.append("title").text((d) => d.title);
            gCell
              .transition()
              .duration(0)
              .delay(delay)
              .attr("visibility", "visible");
            return gCell;
          },
          (update) =>
            update.call((update) =>
              update
                .transition(t)
                .attr("x", (d) => d.x)
                .attr("y", (d) => d.y)
                .attr("width", (d) => d.width)
                .attr("height", (d) => d.height)
                  .end()
                .then(() => {
                  console.log("transition ended");
                  update.remove();
                  callback();
                })
            ),
          (exit) =>
            exit.call((exit) => exit.transition(t).attr("fill", "white").remove()) // fade out
        )
        .attr("class", "cell cell-rect");
      cellElts.sort(frontToBack);      
    }
  }

  // objects is an array of representations
  function visualize(container, objects, lines, duration, animate, rankdir) {
    const reprs = objects;
    const cellPadding = 30;
    const minCellWidth = 30;
    const cellDim = computeCellDim(container, reprs, minCellWidth);
    const cellWidth = cellDim[0];
    const cellHeight = cellDim[1];
    const objectWidths = computeWidths(container, reprs, cellWidth).map(
      (d) => d + cellPadding
    );
    const objectHeights = computeHeights(container, reprs, cellHeight).map(
      (d) => d + cellPadding
    );

    const codeWidths = computeTextWidths(container, lines);
    const maxCodeWidth = Math.max(...codeWidths) + cellPadding;
    const maxCodeHeight = 20; // text height

    const dataExtent = valuesExtent(objects);
    const colour = d3
      .scaleSequential()
      .domain([dataExtent[0], dataExtent[1] * 1.75]) // don't use full upper range
      .interpolator(d3.interpolateBlues);

    // Update the arrays and cells with information for d3
    let xOffset = 0;
    let yOffset = 0;
    if (rankdir === "LR") { // left to right
      yOffset = maxCodeHeight;
    } else { // top to bottom
      xOffset = maxCodeWidth;
    }
    const minChunkSize = 3;
    const codeLines = [];
    for (let i = 0; i < reprs.length; i++) {
      const o = reprs[i];
      for (const cell of arrayElements(o)) {
        if (o.cells) { // array-api
          cell.x = indexToX(cell.index) * cellWidth + xOffset;
          cell.y = indexToY(cell.index, o.shape) * cellHeight + yOffset;
          cell.width = cellWidth * 0.9;
          cell.height = cellHeight * 0.9;
          cell.fill = colour(cell.value);
          cell.title = formatIndex(cell.index);
        } else { // dask
          cell.x = indexToX(cell.offset) * cellWidth + xOffset;
          cell.y = indexToY(cell.offset, o.shape) * cellHeight + yOffset;
          cell.width = Math.max((indexToX(cell.shape) - 0.1) * cellWidth, minChunkSize);
          cell.height = Math.max((indexToY(cell.shape, o.shape) - 0.1) * cellHeight, minChunkSize);
          cell.fill = colour(1);
          // TODO: format indexes as Python tuples
          cell.title = `Chunk\nindex: ${formatIndex(cell.index)}\nshape: ${formatIndex(cell.shape)}\noffset: ${formatIndex(cell.offset)}\nbytes: ${cell.bytes}`;  
        }
      }
      o.x = xOffset;
      o.y = yOffset;
      if (rankdir === "LR") { // left to right
        codeLines.push({
          text: lines[i],
          x: xOffset,
          y: 0
        });  
        xOffset += objectWidths[i];
      } else { // top to bottom
        codeLines.push({
          text: lines[i],
          x: 0,
          y: yOffset
        });  
        yOffset += objectHeights[i];
      }
    }
    let width;
    let height;
    if (rankdir === "LR") { // left to right
      width = objectWidths.reduce((sum, current) => sum + current, 0);
      height = Math.max(...objectHeights) + maxCodeHeight;
    } else {
      width = Math.max(...objectWidths) + maxCodeWidth;
      height = objectHeights.reduce((sum, current) => sum + current, 0);
    }

    // append the svg object to the body of the page
    const svg = d3
      .select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height);

    const g = svg.append("g");

    // make a dictionary of all the cells, keyed by ID
    const cellIdToCell = {};
    for (const o of objects) {
      for (const cell of arrayElements(o)) {
        cellIdToCell[cell.id] = cell;
      }
    }

    function animateArrays(objects, codeLines, index) {
      // Animate every cell in objects[index] from its inputs
      const object = objects[index];

      // Show the code line and the skeleton array
      const group = g.append("g");
      const line = codeLines[index];
      plotCodeAndArray(
        group,
        object,
        line,
        cellWidth,
        cellHeight,
        duration
      );

      // If no animation go to next
      if (!animate || noSources(object)) {
        if (index < objects.length - 1) {
          animateArrays(objects, codeLines, index + 1);
        } else {
          addMouseovers();
          showReloadButton();
        }
        return;
      }

      // Dim other cells during transition
      g.selectAll(".cell").attr("fill-opacity", 0.05);

      // Divide cells into groups that are animated together, using heuristic
      // in animatePerCell to decide if to animate whole array at once - or per-cell
      const cellGroups = animatePerCell(object) ? arrayElements(object).map((elt) => [elt]) : [arrayElements(object)];

      // Each group has a separate delay
      let delay = 0;

      for (let i = 0; i < cellGroups.length; i++) {
        const cellGroup = cellGroups[i];
        const startCells = [];
        const endCells = [];
        for (const cell of cellGroup) {
          if (cell.sources) {
            for (const inputCellId of cell.sources) {
              const startCell = Object.assign({}, cellIdToCell[inputCellId]);
              const endCell = Object.assign({}, cell);
              // create a common key for object constancy
              const commonKey = startCell.id + "_" + endCell.id;
              startCell.key = commonKey;
              endCell.key = commonKey;
              startCells.push(startCell);
              endCells.push(endCell);
            }
          }
        }

        // Now show startCells, and transition to endCells

        const group = g.append("g");

        plotCells(
          group,
          startCells,
          duration,
          delay
        );

        const afterCellGroup = () => {
          g.selectAll(".cell")
            .filter((d) => cellGroup.map((cell) => cell.id).includes(d.id))
            .attr("fill-opacity", 1);
        };

        const afterLastCellGroup = () => {
          // Undim everything
          g.selectAll(".cell").attr("fill-opacity", 1);
          if (index < objects.length - 1) {
            animateArrays(objects, codeLines, index + 1);
          } else {
            addMouseovers();
            showReloadButton();
          }
        };

        plotCells(
          group,
          endCells,
          duration,
          delay,
          i < cellGroups.length - 1 ? afterCellGroup : afterLastCellGroup
        );
        delay += 1000;
      }
    }

    function addMouseovers() {
      g.selectAll(".cell")
        .on("mouseover", (d) => {
          if (d.sources) {
            g.selectAll(".cell").attr("fill-opacity", (d2) => {
              if (d.id == d2.id || d.sources.includes(d2.id)) {
                return 1;
              }
              return 0.05;
            });
          }
        })
        .on("mouseout", (d) => {
          g.selectAll(".cell").attr("fill-opacity", 1);
        });
    }

    function showReloadButton() {
      g.append("g")
        .attr("transform", "translate(0, " + (height - 10) + ")")
        .append("text")
        .attr("class", "reload")
        .text("\u27f3")
        .attr("x", 0)
        .attr("y", 0)
        .on("click", () => {
          g.selectAll("*").remove();
          animateArrays(objects, codeLines, 0);
        });
    }

    animateArrays(objects, codeLines, 0);
  }

  return {
    visualize: visualize
  };
});