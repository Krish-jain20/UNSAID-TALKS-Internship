# Step 1: Get student data
def get_data():
    s_data = {}
    num_s = int(input("Enter the number of students: "))
    subs = ('Math', 'English', 'Science')  

    for _ in range(num_s):
        name = input("\nEnter student name: ")
        marks = {}
        for sub in subs:
            score = int(input(f"Enter marks in {sub}: "))
            marks[sub] = score
        s_data[name] = marks
    return s_data, subs

# Step 2: Calculate average
def calc_avg(m_dict):
    return sum(m_dict.values()) / len(m_dict)

# Step 3: Assign grade
def assign_g(avg):
    if avg >= 90:
        return 'A'
    elif avg >= 75:
        return 'B'
    elif avg >= 60:
        return 'C'
    else:
        return 'D'

# Step 4: Display report for one student
def disp_s_rpt(s_name, s_dict):
    print(f"\nReport for {s_name}")
    marks = s_dict[s_name]
    
    for sub, score in marks.items():
        sub_mark = (sub, score)
        print(f"{sub_mark[0]}: {sub_mark[1]}")
    
    avg = calc_avg(marks)
    grade = assign_g(avg)
    print(f"Average: {avg:.2f}")
    print(f"Grade: {grade}")

# Step 5: Find subject-wise toppers
def find_toppers(s_dict, subs):
    toppers = {}
    for sub in subs:
        max_score = max(s_dict[s][sub] for s in s_dict)
        toppers[sub] = [s for s in s_dict if s_dict[s][sub] == max_score]
    return toppers

# Step 6: Find students scoring above 90 in any subject (using sets)
def find_high_s(s_dict):
    high_s = set()
    for s, marks in s_dict.items():
        if any(score > 90 for score in marks.values()):
            high_s.add(s)
    return high_s

# Step 7: Rank students by total score
def rank_s(s_dict):
    total_scores = {s: sum(marks.values()) for s, marks in s_dict.items()}
    sorted_s = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_s

# Step 8: Calculate subject-wise class averages
def calc_sub_avgs(s_dict, subs):
    avgs = {}
    for sub in subs:
        total = sum(s_dict[s][sub] for s in s_dict)
        avgs[sub] = total / len(s_dict)
    return avgs

# Step 9: Main driver
def main():
    s_data, subs = get_data()

    print("\nCollected Student Data:")
    for s, scores in s_data.items():
        print(f"{s}: {scores}")

    # Print each student’s report
    for s in s_data:
        disp_s_rpt(s, s_data)

    # Show toppers
    toppers = find_toppers(s_data, subs)
    print("\n Subject-wise Toppers:")
    for sub, names in toppers.items():
        print(f"{sub}: {', '.join(names)}")

    # High scorers using sets
    high_s = find_high_s(s_data)
    print("\n Students scoring above 90 in any subject:")
    print(", ".join(high_s) if high_s else "None")

    # Ranking students
    ranked_s = rank_s(s_data)
    print("\n Students Ranked by Total Score:")
    for rank, (s, score) in enumerate(ranked_s, start=1):
        print(f"{rank}. {s} - Total Score: {score}")

    # Subject-wise class averages
    sub_avgs = calc_sub_avgs(s_data, subs)
    print("\n Subject-wise Class Averages:")
    for sub, avg in sub_avgs.items():
        print(f"{sub}: {avg:.2f}")

# Run the program
main()
