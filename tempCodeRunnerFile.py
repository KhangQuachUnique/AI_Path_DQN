route_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]
                m = folium.Map(location=route_coords[0], zoom_start=16)
                folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.7).add_to(m)
                folium.Marker(route_coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
                goal_coords = (graph.nodes[goal_node]['y'], graph.nodes[goal_node]['x'])
                folium.Marker(goal_coords, popup="Goal", icon=folium.Icon(color="red")).add_to(m)
