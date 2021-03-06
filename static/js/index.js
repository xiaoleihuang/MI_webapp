function visualize_word(input_data, selected_id){
    var formt_num_func = d3.format(".3f");
    d3.select(selected_id).selectAll("text")
        .data(input_data)
        .enter().append("text")
        .text(function(d) { return d.word+"  " })
        .style("font-size", function(d) { return (30 ) + "px"; })//+ 30 * parseFloat(d.value)
        .style("font-family", "Impact")
        .style("background-color", function(d) {return d3.interpolateBlues(parseFloat(d.value/3));})
        .attr('data-toggle','tooltip')
        .attr('data-placement','top')
        .attr('title',function(d) { return "The Attention Score is: "+formt_num_func(d.value); });
}
