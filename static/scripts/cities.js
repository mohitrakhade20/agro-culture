var state_arr = new Array('coffee', 'banana', 'pomegranate', 'cotton', 'apple', 'chickpea', 'rice', 'maize', 'blackgram', 'papaya', 'kidneybeans', 'coconut', 'orange', 'watermelon', 'grapes', 'lentil', 'mungbean', 'mango', 'muskmelon', 'pigeonpeas', 'mothbeans', 'jute');

function print_state(state_id){
	// given the id of the <select> tag as function argument, it inserts <option> tags
	var option_str = document.getElementById(state_id);
	option_str.length=0;
	option_str.options[0] = new Option('Select Crop','');
	option_str.selectedIndex = 0;
	for (var i=0; i<state_arr.length; i++) {
		option_str.options[option_str.length] = new Option(state_arr[i],state_arr[i]);
	}
}
